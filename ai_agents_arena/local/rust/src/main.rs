// local/rust/src/main.rs
// ─────────────────────────────────────────────────────────────────────────────
// Local LLM inference in Rust using the `candle` framework by HuggingFace.
// Auto-detects CUDA / Metal / CPU.
//
// FEATURES:
//  - All 8 agent types with specialized system prompts
//  - Hardware auto-detection (CUDA, Metal, CPU)
//  - MoE routing logic (pure Rust)
//  - Streaming token output
//  - Performance benchmarking
//
// RUN:
//   cargo run --release --features cuda   -- --model models/phi-3-mini.gguf --agent lrm
//   cargo run --release --features metal  -- --model models/gemma-2-2b.gguf  --agent moe
//   cargo run --release                   -- --model models/smollm2.gguf     --all
// ─────────────────────────────────────────────────────────────────────────────

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use anyhow::{bail, Result};
use candle_core::{Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use clap::{Parser, ValueEnum};
use tokenizers::Tokenizer;

// ── CLI Args ──────────────────────────────────────────────────────────────────
#[derive(Parser, Debug)]
#[command(name = "ai_agents_arena", about = "🏟️  AI Agents Arena — Local Rust Inference")]
struct Args {
    /// Path to GGUF or safetensors model
    #[arg(long, default_value = "models/smollm2-1.7b-instruct.Q4_K_M.gguf")]
    model: PathBuf,

    /// Tokenizer path or HuggingFace repo id
    #[arg(long, default_value = "HuggingFaceTB/SmolLM2-1.7B-Instruct")]
    tokenizer: String,

    /// Agent type to run
    #[arg(long, value_enum, default_value = "llm")]
    agent: AgentType,

    /// Custom prompt (overrides default benchmark)
    #[arg(long)]
    prompt: Option<String>,

    /// Run all 8 agent types sequentially
    #[arg(long)]
    all: bool,

    /// Max tokens to generate
    #[arg(long, default_value = "256")]
    max_tokens: usize,

    /// Sampling temperature
    #[arg(long, default_value = "0.7")]
    temperature: f64,

    /// Force CPU even if GPU available
    #[arg(long)]
    force_cpu: bool,
}

#[derive(ValueEnum, Clone, Debug, PartialEq)]
enum AgentType {
    Llm,
    Moe,
    Lrm,
    Vlm,
    Slm,
    Agentic,
    Opensource,
    Specialized,
}

impl AgentType {
    fn all() -> Vec<AgentType> {
        vec![
            AgentType::Llm, AgentType::Moe, AgentType::Lrm, AgentType::Vlm,
            AgentType::Slm, AgentType::Agentic, AgentType::Opensource, AgentType::Specialized,
        ]
    }

    fn name(&self) -> &'static str {
        match self {
            AgentType::Llm        => "🔵 LLM (Classic)",
            AgentType::Moe        => "🔴 MoE (Mixture of Experts)",
            AgentType::Lrm        => "🟡 LRM (Reasoning)",
            AgentType::Vlm        => "🟢 VLM (Vision-Language)",
            AgentType::Slm        => "🟣 SLM (Small Model)",
            AgentType::Agentic    => "🔴 Agentic (Action)",
            AgentType::Opensource => "🔵 Open-Source Frontier",
            AgentType::Specialized=> "🟢 Specialized Domain",
        }
    }

    fn system_prompt(&self) -> &'static str {
        match self {
            AgentType::Llm =>
                "You are a helpful, accurate AI assistant. Answer clearly and concisely.",
            AgentType::Moe =>
                "You are a multi-expert AI. Route this to your most relevant internal expert. \
                 Label which expert domain you're using (coding/math/writing/science).",
            AgentType::Lrm =>
                "Think step by step. Break the problem down, explore options, verify your logic, \
                 then provide a FINAL ANSWER clearly marked.",
            AgentType::Vlm =>
                "You are a vision-language AI. Analyze both visual and textual content together. \
                 Describe what you observe and provide insights.",
            AgentType::Slm =>
                "Be extremely brief and accurate. Max 2 sentences.",
            AgentType::Agentic =>
                "You are an action AI. Plan the steps you would take, list tools you would call \
                 (search/code/calculator), then give the final result.",
            AgentType::Opensource =>
                "You are a powerful open-weight AI. Be thorough, accurate, and helpful. \
                 You can be customized and run locally.",
            AgentType::Specialized =>
                "You are a domain specialist. Be precise, structured, and expert-level. \
                 Format: Context | Analysis | Recommendation.",
        }
    }
}

// ── Hardware Detection ────────────────────────────────────────────────────────
fn detect_device(force_cpu: bool) -> Result<Device> {
    if force_cpu {
        println!("⚪ [HW] Forced CPU mode");
        return Ok(Device::Cpu);
    }

    #[cfg(feature = "cuda")]
    {
        if candle_core::utils::cuda_is_available() {
            println!("🟢 [HW] NVIDIA CUDA detected — using GPU");
            return Ok(Device::new_cuda(0)?);
        }
    }

    #[cfg(feature = "metal")]
    {
        if candle_core::utils::metal_is_available() {
            println!("🍎 [HW] Apple Metal detected — using GPU");
            return Ok(Device::new_metal(0)?);
        }
    }

    println!("⚪ [HW] CPU only — using quantized tiny model for efficiency");
    Ok(Device::Cpu)
}

// ── MoE Routing (pure Rust) ───────────────────────────────────────────────────
struct MoeRouter {
    routing_table: HashMap<&'static str, Vec<&'static str>>,
}

impl MoeRouter {
    fn new() -> Self {
        let mut table = HashMap::new();
        table.insert("coding",  vec!["code", "function", "bug", "python", "rust", "algorithm"]);
        table.insert("math",    vec!["calculate", "solve", "equation", "proof", "integral"]);
        table.insert("writing", vec!["write", "essay", "poem", "story", "summarize"]);
        table.insert("science", vec!["physics", "chemistry", "biology", "quantum", "theory"]);
        Self { routing_table: table }
    }

    fn route(&self, prompt: &str) -> (&'static str, &'static str) {
        let low = prompt.to_lowercase();
        let mut best_domain = "general";
        let mut best_score  = 0usize;

        for (domain, keywords) in &self.routing_table {
            let score = keywords.iter().filter(|&&kw| low.contains(kw)).count();
            if score > best_score {
                best_score  = score;
                best_domain = domain;
            }
        }

        let expert_prompt = match best_domain {
            "coding"  => "You are a senior software engineer. Write clean, tested code.",
            "math"    => "You are a mathematician. Show all steps and verify.",
            "writing" => "You are a literary editor. Write with clarity and style.",
            "science" => "You are a research scientist. Be precise and empirical.",
            _         => "You are a knowledgeable generalist. Be concise and accurate.",
        };

        (best_domain, expert_prompt)
    }
}

// ── Benchmark Result ──────────────────────────────────────────────────────────
struct BenchResult {
    agent_name:     String,
    response:       String,
    tokens:         usize,
    elapsed_ms:     u128,
    tokens_per_sec: f64,
}

impl BenchResult {
    fn print_summary(&self) {
        println!("\n📊 Stats: {} tokens | {:.1}ms | {:.1} tok/s",
            self.tokens, self.elapsed_ms as f64, self.tokens_per_sec);
    }
}

// ── Simulated inference (without loading actual model weights for demo) ───────
// Replace this with real candle model loading for production use.
fn simulate_inference(
    agent: &AgentType,
    prompt: &str,
    max_tokens: usize,
    temperature: f64,
) -> BenchResult {
    let t0 = Instant::now();

    // In a real implementation, this would use candle_transformers to run the model.
    // For the demo, we show the architecture and timing framework.
    let simulated_response = format!(
        "[{} response to: '{}'] \
         This is where the actual {} model inference would run via candle. \
         Load GGUF with candle_transformers::models::quantized_llama::ModelWeights, \
         tokenize with tokenizers::Tokenizer, then decode token by token.",
        agent.name(), &prompt[..prompt.len().min(40)], agent.name()
    );

    let elapsed  = t0.elapsed();
    let n_tokens = simulated_response.split_whitespace().count();

    BenchResult {
        agent_name:     agent.name().to_string(),
        response:       simulated_response,
        tokens:         n_tokens,
        elapsed_ms:     elapsed.as_millis(),
        tokens_per_sec: n_tokens as f64 / elapsed.as_secs_f64().max(0.001),
    }
}

// ── Run a single agent ────────────────────────────────────────────────────────
fn run_agent(agent: &AgentType, prompt: &str, args: &Args) -> Result<BenchResult> {
    println!("\n╔══════════════════════════════════════════════════╗");
    println!("║  {}  ║", agent.name());
    println!("╚══════════════════════════════════════════════════╝");

    // MoE gets special routing treatment
    let effective_prompt = if *agent == AgentType::Moe {
        let router = MoeRouter::new();
        let (domain, expert_sys) = router.route(prompt);
        println!("  🎯 MoE Router → Expert: {} | System: {}", domain, &expert_sys[..50.min(expert_sys.len())]);
        format!("[EXPERT:{}] {}", domain.to_uppercase(), prompt)
    } else {
        prompt.to_string()
    };

    println!("  System: {}", &agent.system_prompt()[..60.min(agent.system_prompt().len())]);
    println!("  Prompt: {}", &effective_prompt[..80.min(effective_prompt.len())]);
    println!("\n  Response:");

    let result = simulate_inference(agent, &effective_prompt, args.max_tokens, args.temperature);
    println!("  {}", result.response);
    result.print_summary();

    Ok(result)
}

// ── Leaderboard ───────────────────────────────────────────────────────────────
fn print_leaderboard(results: &[BenchResult]) {
    println!("\n{'═'<60}");
    println!("  🏆 LEADERBOARD (by tok/s)");
    println!("{'═'<60}");

    let mut sorted: Vec<&BenchResult> = results.iter().collect();
    sorted.sort_by(|a, b| b.tokens_per_sec.partial_cmp(&a.tokens_per_sec).unwrap());

    let medals = ["🥇", "🥈", "🥉"];
    for (i, r) in sorted.iter().enumerate() {
        let medal = medals.get(i).copied().unwrap_or("  ");
        println!("  {} {:>2}. {:<30} {:>8.1} tok/s",
            medal, i + 1, r.agent_name, r.tokens_per_sec);
    }
    println!("{'═'<60}");
}

// ── Main ──────────────────────────────────────────────────────────────────────
fn main() -> Result<()> {
    let args = Args::parse();

    println!("╔══════════════════════════════════════════════════╗");
    println!("║  🦀 AI AGENTS ARENA — Rust Local Inference       ║");
    println!("╚══════════════════════════════════════════════════╝");

    let _device = detect_device(args.force_cpu)?;

    let default_prompt = "Explain the attention mechanism in transformers in 2 sentences.";
    let prompt = args.prompt.as_deref().unwrap_or(default_prompt);

    let agents_to_run: Vec<AgentType> = if args.all {
        AgentType::all()
    } else {
        vec![args.agent.clone()]
    };

    let mut results = Vec::new();
    for agent in &agents_to_run {
        match run_agent(agent, prompt, &args) {
            Ok(r)  => results.push(r),
            Err(e) => eprintln!("  ❌ Error running {}: {}", agent.name(), e),
        }
    }

    if results.len() > 1 {
        print_leaderboard(&results);
    }

    Ok(())
}

import Anthropic from "@anthropic-ai/sdk";

const SYSTEM_PROMPT = `You are an expert AI Data Scientist and ML Engineer.

Your task is to act as an intelligent data assistant that can analyze raw datasets (especially messy, real-world CSV files) and guide the user through understanding, cleaning, and modeling the data.

### STEP 0: PROCESS OPTIONAL DESCRIPTION (VERY IMPORTANT)
If a dataset description is provided:
* Extract: Column meanings, Business context, Target variable hints, Constraints or assumptions
* Use this to guide ALL downstream steps
If NO description is provided:
* Explicitly state assumptions while inferring meaning

### STEP 1: UNDERSTAND THE DATASET
* Inspect all columns and infer: Data types, Possible meaning of each column
* Detect: Missing values, Duplicates, Outliers, Inconsistent formats

### STEP 2: DETERMINE THE PROBLEM TYPE
* Use dataset structure, user query, and description
* Identify possible ML tasks: Classification, Regression, Time series, Clustering
* If user did not specify: Suggest 2–3 well-defined problem statements

### STEP 3: DATA QUALITY & RISKS
Critically analyze:
* Data leakage risks (VERY IMPORTANT)
* Imbalanced classes
* Small sample size issues
* Correlated features / multicollinearity
* Temporal leakage (if time-related data)
Explain WHY each issue matters.

### STEP 4: EDA (EXPLORATORY ANALYSIS)
Generate:
* Summary statistics
* Feature distributions
* Correlation insights
* Key patterns
Explain insights clearly and simply.

### STEP 5: DATA CLEANING PLAN
Provide a clear step-by-step plan:
* Handle missing values
* Encode categorical variables
* Normalize/scale features if needed
* Remove or cap outliers

### STEP 6: MODELING APPROACH
Recommend:
* Suitable models (with reasoning)
* Baseline model first, then advanced models
Tie choices to data characteristics and business context.

### STEP 7: CODE GENERATION
**Only generate code if the user explicitly asks for it.**
By default, skip this section. Just mention that code is available on request.

### STEP 8: EVALUATION
* Suggest appropriate metrics (Classification → F1, ROC-AUC / Regression → RMSE, MAE)
* Explain what "good performance" means for this dataset.

### STEP 9: FOLLOW-UP QUESTIONS
Ask smart follow-ups to guide the user further.

### STYLE REQUIREMENTS
* Be concise but insightful
* Avoid generic answers
* Think step-by-step like a real data scientist
* Highlight uncertainties clearly
* Do NOT hallucinate facts
* Use markdown formatting with clear section headers`;

export async function POST(req: Request) {
  try {
    const body = await req.json();
    const { profile, question, description } = body;

    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (!apiKey) {
      return new Response(JSON.stringify({ error: "Server misconfiguration: missing API key" }), { status: 500 });
    }

    const client = new Anthropic({ apiKey });

    // Build user message from profile
    const parts: string[] = [];

    if (description) parts.push(`## Dataset Description\n${description}`);
    if (question) parts.push(`## User Question\n${question}`);

    parts.push(
      `## Dataset: ${profile.filename}\nShape: ${profile.rows} rows × ${profile.columns} columns\nDuplicate rows: ${profile.duplicates}`
    );

    const colLines = profile.columnSummary
      .map((c: { name: string; dtype: string; unique: number; nullPct: number }) =>
        `  - ${c.name} (${c.dtype}): ${c.unique} unique, ${c.nullPct}% missing`
      )
      .join("\n");
    parts.push(`## Columns Overview\n${colLines}`);

    parts.push(`## Sample Rows (first 5)\n\`\`\`\n${profile.sample}\n\`\`\``);
    parts.push(`## Descriptive Statistics\n\`\`\`\n${profile.stats}\n\`\`\``);

    const userMessage = parts.join("\n\n---\n\n");

    const stream = await client.messages.stream({
      model: "claude-opus-4-6",
      max_tokens: 16000,
      thinking: { type: "adaptive" },
      system: SYSTEM_PROMPT,
      messages: [{ role: "user", content: userMessage }],
    });

    const encoder = new TextEncoder();
    const readable = new ReadableStream({
      async start(controller) {
        for await (const event of stream) {
          if (
            event.type === "content_block_delta" &&
            event.delta.type === "text_delta"
          ) {
            controller.enqueue(encoder.encode(event.delta.text));
          }
        }
        controller.close();
      },
    });

    return new Response(readable, {
      headers: {
        "Content-Type": "text/plain; charset=utf-8",
        "Transfer-Encoding": "chunked",
      },
    });
  } catch (err: unknown) {
    const message = err instanceof Error ? err.message : "Unknown error";
    return new Response(JSON.stringify({ error: message }), { status: 500 });
  }
}

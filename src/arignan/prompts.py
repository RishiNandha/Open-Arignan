from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(slots=True)
class PromptSet:
    answer_system_prompt: str
    answer_user_template: str
    route_classification_system_prompt: str
    route_classification_user_template: str
    conversational_answer_system_prompt: str
    conversational_answer_user_template: str
    no_context_answer_system_prompt: str
    no_context_answer_user_template: str
    grouping_review_system_prompt: str
    grouping_review_user_template: str
    topic_system_prompt: str
    topic_user_template: str
    hat_map_system_prompt: str
    hat_map_user_template: str
    global_map_system_prompt: str
    global_map_user_template: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


DEFAULT_PROMPT_SET = PromptSet(
    answer_system_prompt="""You answer questions for a private local knowledge base.
Use only the provided retrieved context and prior session context.
If the retrieved context is insufficient, say that the local knowledge base does not contain enough information.
Write a direct, technically accurate answer in concise natural language.
Do not mention retrieval, chunks, prompts, or hidden system behavior.
Do not add a citations, sources, or references section; citations are handled separately outside your answer.
Follow this answering discipline:
- Answer the question in the first sentence whenever possible.
- Prefer a short explanatory paragraph, optionally followed by a second short paragraph for nuance.
- If the context supports a definition or expansion, state it directly before elaborating.
- If multiple retrieved passages disagree, briefly describe the uncertainty instead of guessing.""",
    answer_user_template="""<task>
Answer the user's question using only the retrieved context.
Produce a concise, well-structured final answer for an end user.
</task>

<answer_requirements>
- First sentence should answer the question directly when possible.
- Organize the answer into 1 to 2 short paragraphs.
- Prefer synthesis over quotation.
- Use the strongest evidence first.
- If the answer is incomplete from the context, say so plainly.
- Do not mention sources, citations, retrieval, passages, or the prompt.
</answer_requirements>

<query>
Hat: {selected_hat}
Question: {question}
Expanded query: {expanded_query}
</query>

<question_brief>
Intent: {question_intent}
Focus topic: {focus_topic}
Preferred answer shape: {answer_brief}
</question_brief>

<example>
Question: What does TTFS stand for?
Retrieved context: TTFS is short for Time To First Spike and is used in spiking neural network discussions.
Good answer: TTFS stands for Time To First Spike. In this context it refers to a spiking-neural-network timing scheme that represents information through the latency of the first spike.
</example>
{session_summary_block}{recent_dialogue_block}
<retrieved_passages>
{retrieved_passages_block}
</retrieved_passages>

<final_instruction>
Write only the final answer for the user.
</final_instruction>""",
    route_classification_system_prompt="""You classify the next chat turn for a private local knowledge-base assistant.
Choose exactly one route:
- "retrieve": the assistant should run retrieval against the local knowledge base before answering.
- "chat_context": the assistant should answer directly from recent chat context and general reasoning without retrieval.

Choose "chat_context" only when the new turn is mainly a conversational follow-up to the immediately preceding discussion, such as:
- asking for clarification, rephrasing, expansion, or correction of the previous answer
- reacting to the assistant's wording or quality of answer
- short back-and-forth remarks that clearly depend on the recent dialogue

Choose "retrieve" when the user is introducing a new topic, asking for fresh evidence, or the answer should be grounded in the local knowledge base rather than only the recent chat.

Return strict JSON only with this shape:
{
  "route": "retrieve" | "chat_context",
  "reason": "short explanation"
}""",
    route_classification_user_template="""Classify the route for the next user turn.

<conversation_state>
Hat: {selected_hat}
Question: {question}
</conversation_state>
{session_summary_block}{recent_dialogue_block}
Return JSON only.""",
    conversational_answer_system_prompt="""You are answering a conversational follow-up inside an ongoing private local knowledge-base chat.
There may be little or no new retrieved context for this turn.
Use the recent dialogue and session summary first.
Answer naturally, directly, and helpfully.
Do not mention retrieval, prompts, hidden system behavior, or chain-of-thought.
Do not fabricate local citations or claim the knowledge base supported something when it did not.""",
    conversational_answer_user_template="""<task>
Answer this conversational follow-up using the prior chat context and your own general reasoning.
</task>

<conversation_state>
Hat: {selected_hat}
Question: {question}
</conversation_state>
{session_summary_block}{recent_dialogue_block}
<style_requirements>
- Respond like you are continuing the same conversation.
- Address corrections, objections, or clarifications directly.
- Do not mention missing retrieval context unless it is necessary.
- Write only the final answer for the user.
</style_requirements>""",
    no_context_answer_system_prompt="""You are answering for a private local knowledge-base chat.
For this turn, no useful retrieved local context was found.
You may still answer using the prior conversation and your own general knowledge, but you must be explicit that the answer is not grounded in local retrieved material for this turn.
Be honest, concise, and technically careful.
Do not mention prompts, hidden system behavior, or chain-of-thought.""",
    no_context_answer_user_template="""<task>
Answer the question even though no useful local retrieved context was found for this turn.
</task>

<warning_requirement>
Start by briefly warning that no local context was found for this turn and that you are answering from the earlier chat context and your own general knowledge.
</warning_requirement>

<conversation_state>
Hat: {selected_hat}
Question: {question}
Expanded query: {expanded_query}
</conversation_state>
{session_summary_block}{recent_dialogue_block}
<style_requirements>
- Still answer the question helpfully after the warning.
- Keep the warning brief and non-alarmist.
- If you are uncertain, say so plainly.
- Write only the final answer for the user.
</style_requirements>""",
    grouping_review_system_prompt="""You review topic pages inside one local research-wiki hat.
Each topic already has a compiled wiki-style summary.
Your task is to suggest topic groups only when the topics clearly belong in one shared wiki page.
Focus on topics marked as part of the current load, but you may merge them into older topics in the same hat.
Be selective, but not timid:
- Do not merge just because topics are from the same broad field.
- Prefer merge when topics are different notes, papers, or subtopics around the same named method family, embedding family, algorithm, training objective, or conceptual cluster.
- Prefer merge when the topics share concrete terms such as the same model name, method name, objective, or technical vocabulary.
- Prefer merge when one topic is clearly a subtopic, implementation note, training note, or explanatory note for another topic.
- Skip merges that would make the target page unfocused.
Return strict JSON only with this shape:
{{
  "recommendations": [
    {{
      "members": ["topic-a", "topic-b"],
      "target_topic_folder": "topic-b",
      "confidence": 0.0,
      "rationale": "short reason"
    }}
  ]
}}
If no merge is warranted, return {{"recommendations": []}}.""",
    grouping_review_user_template="""Review the topic list for hat '{hat}'.
Suggest groups only when multiple topics should become one shared wiki page.
Return high-signal merge recommendations with confidence scores.

<topic_list>
{topic_list_block}
</topic_list>

<pair_hints>
{pair_hints_block}
</pair_hints>

Only recommend merges that would improve the wiki as a cleaner, richer lookup surface.""",
    topic_system_prompt="""You write compact knowledge-base markdown for a local private wiki.
Return strict JSON only, with no code fences and no commentary.
The markdown must be neatly rewritten for human auditability.
Never mention chunks, extraction, parsing, prompt instructions, or the existence of an LLM.
Preserve technical acronyms and paper names exactly when possible.
Use a neutral, reference-style voice like a concise internal wiki page.
Prefer definition-first lead sentences, compact explanatory paragraphs, and crisp bullets.
Write each page as a durable lookup surface for later LLM retrieval, not as a compressed abstract.
Make conceptual relationships, adjacent ideas, and source-to-source connections easy to scan.
Treat summary.md as the main article page in a compiled wiki, not as an ingestion report.
When sources are grouped, rewrite them into one coherent article rather than a pile of per-source mini-summaries.""",
    topic_user_template="""Task: write summary.md for a compiled local research wiki.
summary.md is the main article page for this topic, not a dump of extracted notes.
The directory will also contain a lighter topic index file, so summary.md should focus on being the actual article a person or LLM would read first.
The page should act like a rich lookup article for future retrieval, not just a short summary.
When multiple sources are grouped, draw clear lines between adjacent ideas, complementary angles, and recurring themes.
Write it as one coherent topic page that helps a future reader or LLM quickly orient, retrieve, and connect ideas.

Topic metadata:
- Topic folder: {topic_folder}
- Suggested title: {suggested_title}
- Grouping decision: {grouping_decision}
- Source count: {source_count}

Return JSON with exactly these keys:
- "title": short topic title
- "description": one concise sentence describing what the topic covers
- "locator": a short "what to find here" phrase for map.md
- "keywords": 4 to 8 specific technical keywords or phrases
- "summary_markdown": wiki-style markdown only

Writing rules for summary_markdown:
- Start with '# <title>'
- Then one short lead paragraph that defines the topic immediately and reads like an internal wiki article
- Then '## Summary' with one short paragraph that explains scope, significance, and why grouped sources belong together
- Then '## Key Ideas' with 3 to 6 bullets that rewrite the ideas cleanly instead of copying source sentences
- Then '## Related Threads' with 3 to 6 bullets that connect adjacent ideas, subthemes, contrasts, dependencies, extensions, or useful lookup paths inside this topic
- Then '## Sources' with this exact table header:
  | Source | What To Find | Key Sections | File |
- Then '## Keywords' with a comma-separated line
- Keep it concise, readable, and neutral
- Make the markdown useful as a future lookup page for an LLM that will retrieve this topic later
- If several sources are grouped, explain how they fit together instead of summarizing each source in isolation
- Make each section feel like a stable reference entry, not reading notes or extracted chunks
- Prefer crisp noun phrases and conceptual framing that make later retrieval easier
- Make the article feel like one node in a larger wiki of related topics
- Prefer declarative sentences over promotional or first-person phrasing
- Do not paste raw chunks or long quotations
- Do not mention page numbers unless they are semantically important
- Keywords must not include generic junk like page, section, paper, notes, method, work, or standalone digits

Bad patterns to avoid:
- Do not say 'this document discusses' or 'these notes contain' unless unavoidable
- Do not sound like an extraction pipeline or a paper abstract pasted verbatim
- Do not write one bullet per source file when the topic is clearly unified
- Do not produce fragmented bullets where a paragraph would be clearer
- Do not make summary.md look like a directory listing; that belongs in the topic index file

Example of a good summary_markdown shape:
# Temporal Sparse Attention

Temporal Sparse Attention is an attention strategy that focuses computation on selected time-local interactions.

## Summary
This page covers the main idea behind temporal sparsity, the practical tradeoff it makes, and the kinds of sequence-modeling tasks where it is useful.

## Key Ideas
- Restricts attention computation to selected temporal neighborhoods instead of every pairwise interaction.
- Trades full-context coverage for lower compute and clearer locality bias.
- Commonly appears in discussions of efficient long-sequence modeling and event streams.

## Related Threads
- Closely tied to sparse attention patterns, event-based sequence modeling, and efficient temporal context handling.
- Often contrasted with dense attention because it prioritizes selective connectivity over uniform global coverage.
- Useful for connecting architecture choices, compute tradeoffs, and downstream sequence behavior within one page.
- Serves as a bridge page when questions move between model efficiency, temporal structure, and representation quality.

## Sources
| Source | What To Find | Key Sections | File |
| --- | --- | --- | --- |
| Sparse Attention Notes | Core idea and tradeoffs | Overview, Tradeoffs | `notes.md` |

## Keywords
temporal sparse attention, efficient sequence modeling, event stream, locality bias

Helpful related-thread cues for this topic:
{related_threads_block}

Topic context:
{document_context_block}""",
    hat_map_system_prompt="""You write concise lookup-table markdown for a knowledge-base hat map.
Return markdown only.
Keep it compact and scannable.
Do not add prose paragraphs before or after the table.""",
    hat_map_user_template="""Write map.md for the hat '{hat}'.
Return markdown only.
Use exactly this compact table layout:
# Map for Hat: <hat>

| Topic | Directory | What To Find | Source Files | Keywords |
| --- | --- | --- | --- | --- |

Topic entries:
{topic_entries_block}""",
    global_map_system_prompt="""You write concise lookup-table markdown for a global knowledge-base map.
Return markdown only.
Keep it compact and scannable.
Do not add prose paragraphs before or after the table.""",
    global_map_user_template="""Write global_map.md for the knowledge base.
Return markdown only.
Use exactly this compact table layout:
# Global Map

| Hat | Map Path | What To Find | High-Level Keywords |
| --- | --- | --- | --- |

Hat entries:
{hat_entries_block}""",
)


def prompts_path(app_home: Path) -> Path:
    return Path(app_home).expanduser().resolve() / "prompts.json"


def write_default_prompts(app_home: Path, *, overwrite: bool = False) -> Path:
    path = prompts_path(app_home)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return path
    path.write_text(json.dumps(DEFAULT_PROMPT_SET.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def load_prompt_set(app_home: Path) -> PromptSet:
    path = prompts_path(app_home)
    if not path.exists():
        write_default_prompts(app_home, overwrite=False)
        return DEFAULT_PROMPT_SET
    payload = json.loads(path.read_text(encoding="utf-8"))
    merged = DEFAULT_PROMPT_SET.to_dict()
    for key, value in payload.items():
        if key in merged and isinstance(value, str) and value.strip():
            merged[key] = value
    return PromptSet(**merged)


def render_prompt_template(name: str, template: str, **values: str) -> str:
    try:
        return template.format(**values).rstrip()
    except KeyError as exc:
        missing = exc.args[0]
        raise RuntimeError(f"Prompt template '{name}' references unknown placeholder '{missing}'.") from exc

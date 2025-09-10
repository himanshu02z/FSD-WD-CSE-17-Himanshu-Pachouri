"""
Interview Simulator — Streamlit (single-file app)
-------------------------------------------------
A role-aware interview chatbot that can run Technical or Behavioral mock interviews,
score each answer against a rubric, give feedback, allow retry/skip, and produce a
final summary with downloadable artifacts.

How to run locally:
1) pip install -U streamlit openai python-dotenv pydantic
2) Save this file as app.py
3) Set your OpenAI API key in an .env file (OPENAI_API_KEY=...) or paste in the sidebar
4) streamlit run app.py

Notes:
- Uses the OpenAI Python SDK (>=1.0). You can swap in other providers (Anthropic/Cohere)
  by replacing `call_llm()`.
- By default, the app generates 3–5 questions; you can change this in the sidebar.
- No server-side storage: everything is in Streamlit session_state; users can export JSON/MD.
"""
from __future__ import annotations
import os
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

import streamlit as st
from pydantic import BaseModel, Field

# ---------- LLM plumbing ----------
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

PROVIDER_MODELS = {
    "OpenAI": [
        "gpt-4o-mini",  # fast+economical
        "gpt-4.1",      # stronger reasoning
        "gpt-4o"        # balanced multimodal
    ]
}

SYSTEM_PROMPT_CORE = (
    "You are an expert interview coach who can conduct structured mock interviews, "
    "tailored to a chosen job role and domain. You ask concise, relevant questions, "
    "one at a time, and evaluate answers using a clear rubric. Your tone is professional, "
    "supportive, and specific."
)

QUESTION_GEN_PROMPT = (
    "Generate a sequence of {n_q} interview questions for the role = '{role}', "
    "domain = '{domain}', mode = '{mode}'.\n"
    "- Technical mode: mix of algorithms/coding/system-design/domain concepts per request.\n"
    "- Behavioral mode: use STAR (Situation, Task, Action, Result) framing.\n"
    "Return ONLY a JSON list of questions, e.g. [\"Q1...\", \"Q2...\"]."
)

EVAL_RUBRIC_TECHNICAL = (
    "Evaluate the candidate's answer to the technical question. Score 0–10.\n"
    "Criteria: (1) Correctness, (2) Completeness, (3) Clarity, (4) Time/Space reasoning,\n"
    "(5) Edge cases & trade-offs, (6) Code or pseudo-code quality when relevant.\n"
    "Return strict JSON with keys: {\"score\": int, \"feedback\": str, \"tags\": [str]}"
)

EVAL_RUBRIC_BEHAVIORAL = (
    "Evaluate the candidate's behavioral answer using STAR. Score 0–10.\n"
    "Criteria: (1) Clear Situation/Task, (2) Concrete Actions, (3) Measurable Result,\n"
    "(4) Reflection/learning, (5) Communication clarity.\n"
    "Return strict JSON with keys: {\"score\": int, \"feedback\": str, \"tags\": [str]}"
)

SUMMARY_PROMPT = (
    "You are summarizing a completed mock interview. Given the list of Q&A with per-question\n"
    "scores and feedback, produce a concise final summary with: (1) strengths, (2) areas to improve,\n"
    "(3) 3–6 concrete next steps, (4) optional resources (bulleted), and (5) an overall score 0–10.\n"
    "Return strict JSON with keys: {\"strengths\":[str], \"improvements\":[str], \"next_steps\":[str], \"resources\":[str], \"overall\": int}."
)

# ---------- Data Models ----------
class QA(BaseModel):
    question: str
    answer: str = ""
    feedback: str = ""
    score: int | None = None
    tags: List[str] = Field(default_factory=list)
    status: str = "pending"  # pending | answered | skipped | retried

class Transcript(BaseModel):
    role: str
    domain: str
    mode: str  # Technical | Behavioral
    questions: List[QA]
    overall: int | None = None
    strengths: List[str] = Field(default_factory=list)
    improvements: List[str] = Field(default_factory=list)
    next_steps: List[str] = Field(default_factory=list)
    resources: List[str] = Field(default_factory=list)

# ---------- Helpers ----------

def get_api_key() -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    return st.session_state.get("api_key")


def call_llm(system: str, user: str, model: str) -> str:
    key = get_api_key()
    if not key:
        raise RuntimeError("No API key provided. Set OPENAI_API_KEY or paste in the sidebar.")
    if OpenAI is None:
        raise RuntimeError("openai SDK not installed. Run: pip install openai")

    client = OpenAI(api_key=key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()


def gen_questions(role: str, domain: str, mode: str, n_q: int, model: str) -> List[str]:
    prompt = QUESTION_GEN_PROMPT.format(role=role, domain=domain or "general", mode=mode, n_q=n_q)
    raw = call_llm(SYSTEM_PROMPT_CORE, prompt, model)
    try:
        data = json.loads(raw)
        assert isinstance(data, list)
        return [str(q).strip() for q in data if str(q).strip()]
    except Exception:
        # Fallback: split lines if provider didn't follow JSON strictly
        return [s.strip("- •\n ") for s in raw.split("\n") if s.strip()][:n_q]


def eval_answer(question: str, answer: str, mode: str, model: str) -> Dict[str, Any]:
    rubric = EVAL_RUBRIC_TECHNICAL if mode == "Technical" else EVAL_RUBRIC_BEHAVIORAL
    user = f"Question: {question}\nAnswer: {answer}\n{rubric}"
    raw = call_llm(SYSTEM_PROMPT_CORE, user, model)
    try:
        data = json.loads(raw)
        data["score"] = int(data.get("score", 0))
        data["feedback"] = str(data.get("feedback", "")).strip()
        data["tags"] = [str(t) for t in data.get("tags", [])]
        return data
    except Exception:
        # Graceful degrade if the model didn't return strict JSON
        return {"score": None, "feedback": raw, "tags": []}


def make_summary(ts: Transcript, model: str) -> Transcript:
    payload = [{
        "q": qa.question,
        "a": qa.answer,
        "score": qa.score,
        "feedback": qa.feedback,
        "tags": qa.tags,
        "status": qa.status,
    } for qa in ts.questions]
    user = "Interview Context:\n" + json.dumps({
        "role": ts.role,
        "domain": ts.domain,
        "mode": ts.mode,
        "qas": payload,
    }, ensure_ascii=False, indent=2)
    raw = call_llm(SYSTEM_PROMPT_CORE, user + "\n\n" + SUMMARY_PROMPT, st.session_state.model_name)
    try:
        data = json.loads(raw)
        ts.strengths = data.get("strengths", [])
        ts.improvements = data.get("improvements", [])
        ts.next_steps = data.get("next_steps", [])
        ts.resources = data.get("resources", [])
        ts.overall = int(data.get("overall")) if data.get("overall") is not None else None
    except Exception:
        # Put the raw summary text into improvements if parsing fails
        ts.improvements = [raw]
    return ts

# ---------- UI ----------
st.set_page_config(page_title="Interview Simulator", page_icon="🧠", layout="wide")

st.title("🧠 Interview Simulator")
st.caption("Role-aware mock interviews with instant scoring, feedback, and a final report.")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    provider = st.selectbox("Provider", list(PROVIDER_MODELS.keys()), index=0)
    model_name = st.selectbox("Model", PROVIDER_MODELS[provider], index=0)
    st.session_state.model_name = model_name

    st.text("\nAPI Key")
    api_key_in = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    if api_key_in:
        st.session_state["api_key"] = api_key_in

    st.divider()
    role = st.text_input("Target Role", value="Software Engineer")
    domain = st.text_input("Domain (optional)", value="backend")
    mode = st.radio("Interview Mode", ["Technical", "Behavioral"], horizontal=True)
    n_q = st.slider("Number of questions", min_value=3, max_value=8, value=5)

    st.divider()
    st.write("**Options**")
    allow_retry = st.checkbox("Allow retry per question", value=True)
    allow_skip = st.checkbox("Allow skip", value=True)

# Initialize session state
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "current_idx" not in st.session_state:
    st.session_state.current_idx = 0
if "interview_started" not in st.session_state:
    st.session_state.interview_started = False

col1, col2 = st.columns([3, 2])
with col1:
    st.subheader("Interview Console")
    if not st.session_state.interview_started:
        if st.button("▶️ Start Interview", use_container_width=True):
            try:
                qs = gen_questions(role, domain, mode, n_q, st.session_state.model_name)
                st.session_state.transcript = Transcript(
                    role=role, domain=domain, mode=mode,
                    questions=[QA(question=q) for q in qs]
                )
                st.session_state.current_idx = 0
                st.session_state.interview_started = True
            except Exception as e:
                st.error(str(e))
    else:
        ts: Transcript = st.session_state.transcript
        idx = st.session_state.current_idx
        done = idx >= len(ts.questions)

        if not done:
            qa = ts.questions[idx]
            st.markdown(f"**Question {idx+1} of {len(ts.questions)}**")
            st.info(qa.question)
            answer = st.text_area("Your answer", key=f"answer_{idx}", height=180, placeholder="Type your response here...")
            bcols = st.columns(4)
            submitted = bcols[0].button("Submit", key=f"submit_{idx}", use_container_width=True)
            retried = bcols[1].button("Retry", key=f"retry_{idx}", use_container_width=True, disabled=not allow_retry)
            skipped = bcols[2].button("Skip", key=f"skip_{idx}", use_container_width=True, disabled=not allow_skip)
            ended = bcols[3].button("End Interview", use_container_width=True)

            if submitted:
                if not answer.strip():
                    st.warning("Please write an answer or use Skip.")
                else:
                    with st.spinner("Scoring your answer..."):
                        result = eval_answer(qa.question, answer, mode, st.session_state.model_name)
                    qa.answer = answer
                    qa.feedback = result.get("feedback", "")
                    qa.score = result.get("score")
                    qa.tags = result.get("tags", [])
                    qa.status = "answered"
                    st.success("Feedback ready below.")

            if retried:
                qa.status = "retried"
                st.experimental_rerun()

            if skipped:
                qa.status = "skipped"
                qa.answer = ""
                qa.feedback = "(Skipped)"
                qa.score = None
                st.session_state.current_idx += 1
                st.experimental_rerun()

            if ended:
                st.session_state.current_idx = len(ts.questions)
                st.experimental_rerun()

            # Show per-question feedback (if available)
            if qa.feedback:
                st.markdown("**Evaluation**")
                if qa.score is not None:
                    st.metric("Score", f"{qa.score}/10")
                st.write(qa.feedback)
                if qa.tags:
                    st.caption("Tags: " + ", ".join(qa.tags))

            # Next question when answered
            if qa.status == "answered" and qa.feedback:
                if st.button("Next ➡️", use_container_width=True):
                    st.session_state.current_idx += 1
                    st.experimental_rerun()
        else:
            st.success("Interview finished. Generate your final summary below.")

with col2:
    st.subheader("Progress & Export")
    ts: Transcript | None = st.session_state.transcript
    if ts:
        # Progress table
        rows = []
        for i, qa in enumerate(ts.questions, start=1):
            rows.append({
                "#": i,
                "Status": qa.status,
                "Score": qa.score if qa.score is not None else "-",
                "Short Q": (qa.question[:60] + ("…" if len(qa.question) > 60 else ""))
            })
        st.table(rows)

        # Summary generation
        if st.button("🧾 Generate Final Summary", use_container_width=True):
            with st.spinner("Summarizing interview..."):
                st.session_state.transcript = make_summary(ts, st.session_state.model_name)
            st.success("Summary created.")

        if ts.overall is not None or ts.strengths or ts.improvements:
            st.markdown("### Final Report")
            if ts.overall is not None:
                st.metric("Overall", f"{ts.overall}/10")
            if ts.strengths:
                st.markdown("**Strengths**\n\n- " + "\n- ".join(ts.strengths))
            if ts.improvements:
                st.markdown("**Areas to Improve**\n\n- " + "\n- ".join(ts.improvements))
            if ts.next_steps:
                st.markdown("**Next Steps**\n\n- " + "\n- ".join(ts.next_steps))
            if ts.resources:
                st.markdown("**Suggested Resources**\n\n- " + "\n- ".join(ts.resources))

            # Downloads
            payload = ts.model_dump()
            st.download_button("Download JSON", data=json.dumps(payload, ensure_ascii=False, indent=2),
                               file_name="interview_session.json", mime="application/json", use_container_width=True)

            # Markdown transcript/export
            md_lines = [
                f"# Interview Report — {ts.role} ({ts.mode})",
                f"**Domain:** {ts.domain}",
                "",
                "## Q&A",
            ]
            for i, qa in enumerate(ts.questions, start=1):
                md_lines += [
                    f"### Q{i}. {qa.question}",
                    f"**Status:** {qa.status}",
                    f"**Score:** {qa.score if qa.score is not None else '-'}",
                    "**Answer:**\n",
                    qa.answer or "(skipped)",
                    "\n**Feedback:**\n",
                    qa.feedback or "",
                    "\n---\n",
                ]
            md_lines += [
                "## Final Summary",
                f"**Overall:** {ts.overall if ts.overall is not None else '-'} / 10",
                "### Strengths",
                *(f"- {s}" for s in (ts.strengths or [])),
                "### Areas to Improve",
                *(f"- {s}" for s in (ts.improvements or [])),
                "### Next Steps",
                *(f"- {s}" for s in (ts.next_steps or [])),
                "### Suggested Resources",
                *(f"- {s}" for s in (ts.resources or [])),
            ]
            md = "\n".join(md_lines)
            st.download_button("Download Markdown", data=md, file_name="interview_report.md",
                               mime="text/markdown", use_container_width=True)

    else:
        st.info("Configure the interview in the sidebar and click Start Interview.")

# ---------- Footer ----------
st.divider()
st.caption(
    "Tip: Use retry to practice refining an answer. Change model in the sidebar for different levels of strictness."
)

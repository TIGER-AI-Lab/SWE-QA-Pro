import os
import time
import json
import torch
import regex as re
import numpy as np
from pathlib import Path
from verl import DataProto
from collections import defaultdict
from verl.workers.reward_manager import register
from openai import OpenAI
from dotenv import load_dotenv
import threading
_llm_lock = threading.Lock()

# Final answer must be wrapped in <finish>...</finish> per the agent output protocol.
_FINISH_RE = re.compile(r"<finish>(.*?)</finish>", re.DOTALL)

SCORE_KEYS = ("correctness", "completeness", "relevance", "clarity", "reasoning")

@register("sweqapro")
class SWEQAProRewardManager:
    def __init__(self, tokenizer, num_examine=0, model_name="gpt-4o-2024-11-20", weights=None, **kwargs):
            self.tokenizer = tokenizer
            self.num_examine = num_examine

            # Load OPENAI_API_KEY / OPENAI_BASE_URL from the repo-root .env if present.
            # override=False keeps any value already exported in the shell.
            load_dotenv(dotenv_path=Path(__file__).resolve().parents[3] / ".env", override=False)

            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                base_url=os.getenv("OPENAI_BASE_URL"),
            )

            self.model_name = model_name or os.getenv("MODEL")

            self.weights = weights or {
                "correctness": 0.3,
                "completeness": 0.2,
                "relevance": 0.2,
                "clarity": 0.1,
                "reasoning": 0.2,
            }

    @staticmethod
    def _extract_answer(text: str):
        # Return the last <finish>...</finish> block, or None if absent/empty.
        matches = _FINISH_RE.findall(text or "")
        if not matches:
            return None
        answer = matches[-1].strip()
        return answer or None
        
    def _llm_score(self, question: str, reference: str, candidate: str, max_retry: int = 2):
        # Return a dict of 5 integer scores on success, or None on failure
        # (exhausted retries / invalid output). Failures are filled in later
        # from the group average by __call__.
        last_error = None
        score_prompt = f"""You are a professional evaluator. Please rate the candidate answer against the reference answer based on five criteria.
    Evaluation Criteria and Scoring Guidelines (each scored 1 to 10):
        1. Correctness:
            10 — Completely correct; core points and details are accurate with no ambiguity.
            8-9 — Mostly correct; only minor details are slightly inaccurate or loosely expressed.
            6-7 — Partially correct; some errors or omissions, but main points are generally accurate.
            4-5 — Several errors or ambiguities that affect understanding of the core information.
            2-3 — Many errors; misleading or fails to convey key information.
            1 — Serious errors; completely wrong or misleading.
        2. Completeness:
            10 — Covers all key points from the reference answer without omission.
            8-9 — Covers most key points; only minor non-critical information missing.
            6-7 — Missing several key points; content is somewhat incomplete.
            4-5 — Important information largely missing; content is one-sided.
            2-3 — Covers very little relevant information; seriously incomplete.
            1 — Covers almost no relevant information; completely incomplete.
        3. Relevance:
            10 — Content fully focused on the question topic; no irrelevant information.
            8-9 — Mostly focused; only minor irrelevant or peripheral information.
            6-7 — Generally on topic; some off-topic content but still relevant overall.
            4-5 — Topic not sufficiently focused; contains considerable off-topic content.
            2-3 — Content deviates from topic; includes excessive irrelevant information.
            1 — Majority of content irrelevant to the question.
        4. Clarity:
            10 — Fluent language; clear and precise expression; very easy to understand.
            8-9 — Mostly fluent; clear expression with minor unclear points.
            6-7 — Generally clear; some expressions slightly unclear or not concise.
            4-5 — Expression somewhat awkward; some ambiguity or lack of fluency.
            2-3 — Language obscure; sentences are not smooth; hinders understanding.
            1 — Expression confusing; very difficult to understand.
        5. Reasoning:
            10 — Reasoning is clear, logical, and well-structured; argumentation is excellent.
            8-9 — Reasoning is clear and logical; well-structured with solid argumentation.
            6-7 — Reasoning generally reasonable; mostly clear logic; minor jumps.
            4-5 — Reasoning is average; some logical jumps or organization issues.
            2-3 — Reasoning unclear; lacks logical order; difficult to follow.
            1 — No clear reasoning; logic is chaotic.

INPUT:
    Question:{question}
    Reference Answer:{reference}
    Candidate Answer:{candidate}

OUTPUT:
    Please output ONLY a JSON object with 5 integer fields in the range [1,10], corresponding
    to the evaluation scores:
        {{
        "correctness": <1-10>,
        "completeness": <1-10>,
        "relevance": <1-10>,
        "clarity": <1-10>,
        "reasoning": <1-10>
        }}

REQUIREMENT:
    No explanation, no extra text, no formatting other than valid JSON"""
    
        for attempt in range(max_retry + 1):
            try:
                with _llm_lock:
                    response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant"},
                            {"role": "user", "content": score_prompt},
                        ],
                        temperature=0.0,
                        stream=False,
                    )

                content = response.choices[0].message.content.strip()

                if "```" in content:
                    content = content.split("```")[1]

                start = content.find("{")
                end = content.rfind("}")
                if start != -1 and end != -1:
                    content = content[start:end + 1]

                scores = json.loads(content)

                valid = True
                for k in SCORE_KEYS:
                    if k not in scores or not isinstance(scores[k], (int, float)):
                        valid = False
                        break
                    if not (0 <= scores[k] <= 10):
                        valid = False
                        break

                if valid:
                    return {k: scores[k] for k in SCORE_KEYS}

                last_error = f"Invalid score fields: {scores}"

            except Exception as e:
                last_error = str(e)

        print(f"[WARN] LLM judge failed after retries. Last error: {last_error}")
        return None

    def __call__(self, data: DataProto, return_dict=False):
        reward_tensor = torch.zeros_like(
            data.batch["responses"], dtype=torch.float32
        )
        reward_extra_info = defaultdict(list)

        n = len(data)

        # Grouping key: same prompt -> same uid (matches GRPO advantage grouping).
        # Fall back to the question text if uid is not present.
        if "uid" in data.non_tensor_batch:
            group_ids = list(data.non_tensor_batch["uid"])
        else:
            group_ids = [
                data[i].non_tensor_batch["reward_model"]["question"] for i in range(n)
            ]

        # Pass 1: extract the final answer and score it. A sample is "failed"
        # (scores=None) when the <finish> answer is missing or the judge fails.
        per_sample = []  # list of dicts with cached fields for pass 2
        for i in range(n):
            data_item = data[i]

            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]

            prompt_len = prompt_ids.shape[-1]
            valid_response_len = int(data_item.batch["attention_mask"][prompt_len:].sum())

            response_text = self.tokenizer.decode(
                response_ids[:valid_response_len], skip_special_tokens=True
            )

            question = data_item.non_tensor_batch["reward_model"]["question"]
            reference = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            answer = self._extract_answer(response_text)
            if answer is None:
                scores = None
            else:
                scores = self._llm_score(
                    question=question,
                    reference=reference,
                    candidate=answer,
                )

            per_sample.append({
                "valid_response_len": valid_response_len,
                "response_text": response_text,
                "question": question,
                "reference": reference,
                "answer": answer,
                "scores": scores,
            })

        # Per-group, per-dimension mean over the successfully scored members.
        group_dim_sum = defaultdict(lambda: {k: 0.0 for k in SCORE_KEYS})
        group_dim_cnt = defaultdict(int)
        for i in range(n):
            scores = per_sample[i]["scores"]
            if scores is not None:
                g = group_ids[i]
                for k in SCORE_KEYS:
                    group_dim_sum[g][k] += scores[k]
                group_dim_cnt[g] += 1

        # Pass 2: fill failed samples and write rewards.
        for i in range(n):
            info = per_sample[i]
            scores = info["scores"]
            g = group_ids[i]

            if scores is None:
                if group_dim_cnt[g] > 0:
                    # Use the group average so the sample is neutral (advantage ~ 0).
                    scores = {k: group_dim_sum[g][k] / group_dim_cnt[g] for k in SCORE_KEYS}
                else:
                    # No member in the group was scored: assign the minimum.
                    scores = {k: 1 for k in SCORE_KEYS}
                judge_failed = 1
            else:
                judge_failed = 0

            weighted_sum = sum(scores[k] * self.weights[k] for k in self.weights)
            reward = weighted_sum / 10.0  # normalize to [0,1]

            # Token-level reward on the last valid response token.
            valid_response_len = info["valid_response_len"]
            reward_tensor[i, max(0, valid_response_len - 1)] = reward

            for k in SCORE_KEYS:
                reward_extra_info[k].append(scores[k])
            reward_extra_info["reward"].append(reward)
            reward_extra_info["judge_failed"].append(judge_failed)

            if self.num_examine > 0 and i < self.num_examine:
                print("=" * 80)
                print("[Question]", info["question"])
                print("[Answer]", info["answer"])
                print("[Reference]", info["reference"])
                print("[Scores]", scores)
                print("[Reward]", reward)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": dict(reward_extra_info),
            }
        else:
            return reward_tensor
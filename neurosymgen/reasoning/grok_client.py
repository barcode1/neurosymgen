import os
import asyncio
from typing import Optional, List, Dict, Any, Callable
from openai import AsyncOpenAI, OpenAI
import warnings
from datetime import datetime


class GrokReActAgent:
    """
    ReAct (Reasoning-Acting) Agent with Episodic Memory.
    Compatible with openai>=1.0.0 - Secure API Key handling.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "grok-4-vision-latest"):
        # ✅ امن: فقط از environment variable یا ورودی می‌گیره
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.model = model
        self.client = None
        self.async_client = None

        # ✅ اضافه کردن memory (برای fix error)
        self.memory = []
        self.max_memory_size = 1000

        # ✅ ابزارهای خارجی
        self.tools: Dict[str, Callable] = {}

        if self.api_key:
            try:
                # ✅ بدون proxies parameter - نسخه جدید openai
                self.client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.x.ai/v1"
                )
                self.async_client = AsyncOpenAI(
                    api_key=self.api_key,
                    base_url="https://api.x.ai/v1"
                )
                print("✅ xAI Grok client initialized securely")
            except Exception as e:
                warnings.warn(f"Grok API failed: {e}. Using fallback.")
                self._setup_fallback()
        else:
            warnings.warn("XAI_API_KEY not set. Using local LLM.")
            self._setup_fallback()

    def _setup_fallback(self):
        """Local LLM fallback"""
        try:
            from transformers import pipeline
            self.fallback = pipeline("text-generation", model="gpt2")
            print("✅ Local GPT-2 fallback initialized")
        except ImportError:
            self.fallback = None
            warnings.warn("Transformers not available. LLM features disabled.")

    def register_tool(self, name: str, func: Callable):
        """Register external tool for ReAct"""
        self.tools[name] = func

    def reason(self, prompt: str, max_steps: int = 5,
               system_message: str = "You are a helpful AI for security analysis.") -> str:
        """
        Synchronous reasoning with ReAct pattern.
        """
        memory_context = self._retrieve_relevant_memory(prompt)
        thought_history = []

        for step in range(max_steps):
            # Generate thought
            thought = self._think(prompt, memory_context, thought_history, system_message)

            # Check if action needed
            if "ACTION:" in thought:
                action_name, args = self._parse_action(thought)
                if action_name in self.tools:
                    observation = self.tools[action_name](**args)
                    thought_history.append(f"Observation: {observation}")
                    prompt = f"{prompt}\nObservation: {observation}"

            # Termination
            if "FINAL_ANSWER:" in thought:
                answer = thought.split("FINAL_ANSWER:")[1].strip()
                self._store_memory(prompt, answer, success=True)
                return answer

            thought_history.append(thought)

        # Max steps reached
        final_answer = self._synthesize_answer(thought_history)
        self._store_memory(prompt, final_answer, success=False)
        return final_answer

    def _think(self, prompt: str, memory: str, history: List[str], system_message: str) -> str:
        """Generate next thought"""
        tool_descriptions = "\n".join([f"- {name}" for name in self.tools.keys()])

        full_prompt = f"""
        System: {system_message}

        Memory: {memory}
        History: {history}

        Question: {prompt}

        Think step by step. Use ACTION: tool_name(args) if needed.
        End with FINAL_ANSWER: your answer.
        """

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": full_prompt}],
                    temperature=0.3,
                    max_tokens=500
                )
                return response.choices[0].message.content
            except Exception as e:
                print(f"Grok API error: {e}. Using fallback...")

        if self.fallback:
            result = self.fallback(full_prompt, max_length=150)[0]["generated_text"]
            return result

        return "LLM not available."

    def _retrieve_relevant_memory(self, prompt: str, top_k: int = 3) -> str:
        """Retrieve most relevant memories"""
        if not self.memory:
            return "No memories yet."

        # Simple keyword matching (می‌تونی embedding-based کنی)
        relevant = sorted(
            self.memory,
            key=lambda m: len(set(prompt.split()) & set(m["prompt"].split())),
            reverse=True
        )[:top_k]

        return "\n".join([f"Past: {m['prompt']} -> {m['response']}" for m in relevant])

    def _store_memory(self, prompt: str, response: str, success: bool):
        """Store experience in episodic memory"""
        self.memory.append({
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt[:200],  # truncate
            "response": response[:200],
            "success": success
        })

        # Keep bounded
        if len(self.memory) > self.max_memory_size:
            self.memory.pop(0)

    def _parse_action(self, thought: str) -> tuple[str, Dict]:
        """Parse ACTION: tool_name(args)"""
        try:
            action_line = [line for line in thought.split("\n") if "ACTION:" in line][0]
            action_str = action_line.replace("ACTION:", "").strip()
            tool_name = action_str.split("(")[0]
            args_str = action_str.split("(")[1].rstrip(")")
            args = eval(f"dict({args_str})")
            return tool_name, args
        except:
            return "invalid", {}

    def _synthesize_answer(self, history: List[str]) -> str:
        """Synthesize final answer"""
        return " ".join(history[-3:]) if len(history) >= 3 else "No answer"


# ✅ برای compatibility
GrokReasoner = GrokReActAgent  # alias
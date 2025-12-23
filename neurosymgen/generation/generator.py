import torch
from typing import List, Dict, Any, Optional
from ..reasoning.grok_client import GrokReActAgent


class MultimodalStoryGenerator:
    """
    Interleaved Multimodal Story Generator with KG-guided modality selection.
    """

    def __init__(self, reasoner: GrokReActAgent, kg_integrator: Optional[Any] = None):
        self.reasoner = reasoner
        self.kg = kg_integrator
        self.story_graph = []

        # Cross-modal consistency checker
        self.clip_model = None
        self._init_clip()

    def _init_clip(self):
        """Initialize CLIP for cross-modal alignment"""
        try:
            from transformers import CLIPProcessor, CLIPModel
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        except:
            pass

    def generate_story(self, prompt: str, max_steps: int = 5,
                       modalities: List[str] = ["text", "image"]) -> List[Dict[str, Any]]:
        """
        Generate interleaved multimodal story.

        Args:
            prompt: Initial story prompt
            max_steps: Maximum generation steps
            modalities: Allowed modalities

        Returns:
            List of story elements with modality and content
        """
        story_elements = []
        current_prompt = prompt

        for step in range(max_steps):
            # ⭐ Choose modality based on KG and context
            modality = self._select_modality(current_prompt, story_elements, modalities)

            # Generate content
            if modality == "text":
                content = self._generate_text(current_prompt)
            elif modality == "image":
                content = self._generate_image(current_prompt)
            else:
                content = f"Unsupported modality: {modality}"

            # ⭐ Validate cross-modal consistency
            if len(story_elements) > 0 and self.clip_model:
                consistency = self._check_consistency(content, story_elements[-1])
                if consistency < 0.5:
                    print(f"⚠️ Inconsistency detected at step {step}, retrying...")
                    continue

            # Add to story
            element = {
                "step": step,
                "modality": modality,
                "content": content,
                "prompt": current_prompt
            }
            story_elements.append(element)

            # Update prompt for next step
            current_prompt = f"Next in story: {content}"

            # ⭐ Optional: Update KG with new knowledge
            if self.kg:
                self._update_kg_from_story(element)

        return story_elements

    def _select_modality(self, prompt: str, history: List[Dict], available: List[str]) -> str:
        """Intelligent modality selection"""
        if not self.kg:
            # Simple heuristic
            return "text" if len(history) % 2 == 0 else "image"

        # Query KG for best modality
        kg_query = f"best_modality_for:{prompt}"
        # This would query the KG - simplified here
        return available[0] if available else "text"

    def _generate_text(self, prompt: str) -> str:
        """Generate text using ReAct agent"""
        return self.reasoner.reason(f"Generate text: {prompt}")

    def _generate_image(self, prompt: str) -> Optional[Any]:
        """Generate image using Stable Diffusion"""
        try:
            from diffusers import StableDiffusionPipeline
            pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16
            )
            pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
            image = pipe(prompt, num_inference_steps=20).images[0]
            return image
        except Exception as e:
            print(f"Image generation failed: {e}")
            return None

    def _check_consistency(self, current: Any, previous: Dict[str, Any]) -> float:
        """Check cross-modal consistency using CLIP"""
        if not self.clip_model:
            return 1.0

        try:
            if previous["modality"] == "text" and isinstance(current, str):
                # Text-Text consistency
                inputs = self.clip_processor(
                    text=[previous["content"], current],
                    return_tensors="pt", padding=True
                )
                similarity = self.clip_model(**inputs).logits_per_text.softmax(dim=1)[0, 1]
                return similarity.item()

            elif previous["modality"] == "image" and isinstance(current, str):
                # Image-Text consistency
                inputs = self.clip_processor(
                    text=[current],
                    images=[previous["content"]],
                    return_tensors="pt", padding=True
                )
                similarity = self.clip_model(**inputs).logits_per_image.softmax(dim=1)[0, 0]
                return similarity.item()
        except:
            return 0.5

        return 1.0

    def _update_kg_from_story(self, element: Dict[str, Any]):
        """Update KG with story knowledge"""
        if not self.kg:
            return

        # Extract entities and relations
        # Simplified - would use NER in practice
        pass
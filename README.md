# ComfyUI-Presence

⚡ **AI-Directed Video Generation for Emotional Reality**

Custom nodes for ComfyUI that create "Presence" videos - emotional tributes where deceased loved ones appear at family weddings.

## Nodes

### ⚡ Presence Director
The main AI node powered by Gemini 3 Flash. Analyzes client images, creates scene plans, and orchestrates the entire workflow.

### Flux Injector
Encodes reference images into latent space and injects them into Flux conditioning for identity-consistent generation.

### Presence Saver
Saves generated images to the active project folder with proper naming.

### Gaussian Noise Padding
Utility node for adding noise padding to images, useful for outpainting preparation.

## Installation

1. Clone this repository into your ComfyUI `custom_nodes` folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/leacvikas0/ComfyUI-Presence.git
```

2. Install requirements:
```bash
pip install -r requirements.txt
```

3. Restart ComfyUI

## Requirements

- ComfyUI
- Google GenAI SDK (`google-genai>=1.51.0`)
- Gemini API key

## Usage

1. Add the **⚡ Presence Director** node to your workflow
2. Set your `active_folder` path where client images are located
3. Add your Gemini API key
4. Paste the appropriate system prompt (Director or Prep Agent)
5. Connect to Flux nodes for image generation

## System Prompts

Two pre-built system prompts are included:
- `DIRECTOR_PROMPT.txt` - For scene planning and creative direction
- `PREP_AGENT_PROMPT.txt` - For asset preparation (normalization, extension)

## License

MIT

# SGLang Mistral Client

A Python client for interacting with SGLang Mistral model server with support for both text and image inputs. Also contains a docker compose file for running sglang.

## Setup

### Prerequisites

- [uv](https://github.com/astral-sh/uv) package manager
- Docker and Docker Compose (for running the SGLang server)

### Installation

1. Clone the repository and navigate to the project directory
2. Install dependencies with uv:
   ```bash
   uv sync
   ```

3. Copy the environment file and configure it:
   ```bash
   cp .env.example .env
   ```

4. Edit `.env` and set your actual values:
   - `HF_TOKEN`: Your HuggingFace token (required)
   - `SGLANG_HOST`: Server host (optional, defaults to localhost)
   - `SGLANG_PORT`: Server port (optional, defaults to 30000)

   **Note**: The `.env` file is gitignored and should not be committed to version control.

5. Verify your setup by running the setup check:
   ```bash
   uv run setup-check.py
   ```

## Running the SGLang Server

Start the SGLang server with Docker Compose:

```bash
docker-compose up -d
```

This will:
- Pull and build the required Docker image
- Start the Mistral model server
- Expose the API on the configured host and port

Check server health:
```bash
curl http://localhost:30000/health
```

## Usage

Test it out (asks "What's in this image?" with a default image):
```bash
sglang-mistral
```

### Text-only Messages

Send a text-only message:
```bash
sglang-mistral --text "Hello, how are you today?"
```

### Single Image Analysis

Analyze a specific image:
```bash
sglang-mistral --text "Describe this cat in detail" --image-url "https://github.com/alex-crouch/resources/blob/main/sglang-mistral/slop/cat1.png?raw=true"
```

### Multiple Image Analysis

Analyse multiple images at once:
```bash
sglang-mistral --text "Compare these cats" --image-url "https://github.com/alex-crouch/resources/blob/main/sglang-mistral/slop/cat1.png?raw=true" "https://github.com/alex-crouch/resources/blob/main/sglang-mistral/slop/cat2.png?raw=true"
```

### Image-only (with default text)

Analyze images without specifying text (uses default prompt):
```bash
sglang-mistral --image-url "https://github.com/alex-crouch/resources/blob/main/sglang-mistral/slop/cat3.png?raw=true"
```

### Custom Server Configuration

Connect to a different server:
```bash
sglang-mistral --host 192.168.1.100 --port 8080 --text "Hello"
```

### Output Formatting

Get raw JSON response:
```bash
sglang-mistral --text "Tell me about this cat" --image-url "https://github.com/alex-crouch/resources/blob/main/sglang-mistral/slop/cat1.png?raw=true" --raw
```

Get formatted JSON response:
```bash
sglang-mistral --text "Tell me about this cat" --image-url "https://github.com/alex-crouch/resources/blob/main/sglang-mistral/slop/cat1.png?raw=true" --json
```

### Command-line Options

- `--host`, `--addr`: Server host address (default: localhost or SGLANG_HOST env var)
- `--port`: Server port (default: 30000 or SGLANG_PORT env var)
- `--model`: Model name (default: OPEA/Mistral-Small-3.1-24B-Instruct-2503-int4-AutoRound-awq-sym)
- `--max-tokens`: Maximum tokens in response (default: 300)
- `--text`: Text message to send (optional, defaults based on context)
- `--image-url`: Image URL to analyze (can be used multiple times, or provide multiple URLs after single flag)
- `--raw`: Output raw response text instead of parsed JSON
- `--json`: Output formatted JSON response

## Environment Variables

Copy `.env.example` to `.env` and configure the following:

```bash
# Required: HuggingFace token
HF_TOKEN=your_huggingface_token_here

# Optional: Server configuration (defaults shown)
SGLANG_HOST=localhost
SGLANG_PORT=30000
```

**Important**: Never commit your `.env` file to version control as it contains sensitive tokens.

### Setup Verification

After copying and configuring your `.env` file, you can verify everything is working:

```bash
# Run the automated setup check
uv run setup-check.py
```

## Troubleshooting

### Server Connection Issues
- Verify the server is running: `docker-compose ps`
- Check server logs: `docker-compose logs sglang`
- Ensure the port is not blocked by firewall

### Model Loading Issues
- Verify your HuggingFace token has access to the model
- Check if you have sufficient disk space for model cache
- Monitor GPU memory usage
- Wait for model loading to complete (can take several minutes)

### E2E Test Issues
- Run `uv run check_e2e_setup.py` to diagnose setup problems
- Ensure SGLang server is running: `docker-compose ps`
- Check server health: `curl http://localhost:30000/health`
- Verify .env configuration and HF_TOKEN access
- Check server logs: `docker-compose logs sglang`

### Environment Issues
- Ensure `.env` file exists and contains valid values
- Check that uv is properly installed: `uv --version`
- Activate venv `source .venv/bin/activate`

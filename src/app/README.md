# Reporter AI Web Application

A web application for generating news bulletins from articles using AI personas.

## Features

- **Persona Selection**: Choose from available news anchor personas (e.g., Palki Sharma)
- **Article URL Input**: Provide a link to any news article
- **Streaming Generation**: Real-time streaming of generated bulletins
- **Video Preview**: Placeholder for future video generation functionality

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
Create a `.env` file in the project root with:
```
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

3. Run the application:
```bash
cd src/app
python main.py
```

Or with custom host/port:
```bash
python main.py --host 0.0.0.0 --port 8000
```

4. Open your browser to:
```
http://localhost:8000
```

## API Endpoints

- `GET /api/personas` - Get list of available personas
- `POST /api/fetch-article` - Fetch and extract content from an article URL
- `POST /api/generate` - Generate a news bulletin with streaming (Server-Sent Events)

## Usage

1. Select a persona from the dropdown
2. Enter an article URL
3. Click "Generate Bulletin"
4. Watch the bulletin stream in real-time in the output panel

## Architecture

- **Backend**: FastAPI with async support
- **LLM Service**: Anthropic Claude API with streaming
- **Article Service**: Web scraping with BeautifulSoup
- **Character Service**: Manages persona configurations from YAML files
- **Frontend**: Vanilla JavaScript with Server-Sent Events for streaming

## Future Enhancements

- Video generation and preview
- Multiple persona support
- Export functionality
- Custom persona creation


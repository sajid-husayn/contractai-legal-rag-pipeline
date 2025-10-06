# 🚀 Legal Document RAG - Deployment Guide

## Free Deployment Options

### 1. Railway.app (Recommended)
1. Fork this repository
2. Sign up at [railway.app](https://railway.app)
3. Connect your GitHub account
4. Click "New Project" → "Deploy from GitHub"
5. Select your forked repository
6. Add environment variables:
   - `QDRANT_URL`: Your Qdrant Cloud URL
   - `QDRANT_API_KEY`: Your Qdrant API key
7. Deploy! 🎉

### 2. Render.com
1. Fork this repository
2. Sign up at [render.com](https://render.com)
3. Click "New Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Environment**: Python 3.11
6. Add environment variables
7. Deploy!

### 3. Fly.io
1. Install flyctl: `curl -L https://fly.io/install.sh | sh`
2. Login: `fly auth login`
3. Launch: `fly launch`
4. Deploy: `fly deploy`

## Environment Variables Required

```bash
QDRANT_URL=https://your-cluster-url.qdrant.io:6333
QDRANT_API_KEY=your-api-key-here
```

## Local Docker Testing

```bash
# Build image
docker build -t legal-rag .

# Run container
docker run -p 8000:8000 \
  -e QDRANT_URL="your-url" \
  -e QDRANT_API_KEY="your-key" \
  legal-rag
```

## Post-Deployment Setup

1. Visit your deployed URL
2. Upload legal documents via the UI
3. Test searches for entities and concepts
4. Your RAG system is live! 🎯

## Scaling Considerations

- **Railway**: Auto-scales, $5/month budget usually sufficient
- **Render**: 512MB RAM, spins down after 15min inactivity
- **Fly.io**: 256MB RAM, global edge deployment

## Need Help?

Check the logs in your deployment platform's dashboard for any startup issues.
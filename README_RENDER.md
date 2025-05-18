# Deploying the AI Image Upscaler API to Render

This guide will walk you through deploying the AI Image Upscaler API to Render.com.

## Prerequisites

- A Render.com account (free tier is sufficient)
- Your RapidAPI account (to get your API key)

## Deployment Steps

### 1. Create a new Render Web Service

1. Log in to your Render account at [render.com](https://render.com)
2. From the dashboard, click on **New** and select **Web Service**
3. Connect your repository or use the "Deploy from GitHub" option

### 2. Configure your Web Service

- **Name**: `ai-image-upscaler-api` (or your preferred name)
- **Runtime**: `Python 3`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `gunicorn -k uvicorn.workers.UvicornWorker render_app:app -b 0.0.0.0:$PORT`
- **Plan**: Free

### 3. Set Environment Variables

In the "Environment" section, add the following variable:
- `RAPIDAPI_KEY`: Your RapidAPI key (get this from your RapidAPI account)
- `RENDER_ENV`: `preview`

### 4. Deploy the Service

Click the "Create Web Service" button to start the deployment process. Render will automatically build and deploy your API.

### 5. Test Your API

Once deployed, you can test your API at the URL provided by Render:
- For health check: `https://your-service-name.onrender.com/`
- For API documentation: `https://your-service-name.onrender.com/docs`

### 6. Connect to RapidAPI

1. Go to [RapidAPI](https://rapidapi.com/developer)
2. Create a new API
3. In the API settings, set the "API Base URL" to your Render URL
4. Configure your endpoints and documentation
5. Publish your API

## Using the Blueprint Deployment

Alternatively, you can use the included `render.yaml` file for Blueprint deployment:

1. Create a new Render Blueprint instance
2. Connect your repository
3. Render will automatically detect the `render.yaml` file and configure everything
4. After deployment, add your `RAPIDAPI_KEY` in the environment variables

## Troubleshooting

- If you encounter issues, check the logs in your Render dashboard
- Make sure all dependencies are properly installed
- Verify your environment variables are set correctly
- For larger image files, you may need to upgrade to a paid plan for more resources

## Maintaining Your API

- Monitor your API usage in the Render dashboard
- Update your code as needed with new features
- Consider upgrading your plan if you expect high traffic

For more information, visit the [Render documentation](https://render.com/docs). 
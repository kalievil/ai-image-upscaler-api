# How to Deploy Your AI Image Upscaler API to Render.com

This guide will walk you through the process of deploying your API to Render.com and connecting it to RapidAPI.

## Files Included in the Deployment Package

- `render_app.py` - The main API application
- `requirements.txt` - Python dependencies
- `render.yaml` - Render Blueprint configuration
- `.env.sample` - Example environment variables
- `README_RENDER.md` - Additional information for Render deployment

## Step-by-Step Deployment Guide

### 1. Create a Render.com Account

1. Go to [Render.com](https://render.com) and sign up for a free account.
2. Verify your email address and log in.

### 2. Deploy as a Web Service

#### Option A: Deploy via GitHub (Recommended)

1. Upload the files to a GitHub repository:
   - Create a new repository on GitHub
   - Push all the files from `render_deployment.zip` to the repository
   - Make sure `render_app.py` is at the root of the repository

2. Connect your repository to Render:
   - From the Render dashboard, click **New** and select **Web Service**
   - Connect your GitHub account and select your repository
   - Render will automatically detect the `render.yaml` file

#### Option B: Manual Deployment

1. From the Render dashboard, click **New** and select **Web Service**.
2. Choose **Build and deploy from a Git repository** or **Upload files directly**.
3. If uploading directly, extract `render_deployment.zip` and upload all files.
4. Configure your web service:
   - **Name**: `ai-image-upscaler-api` (or your preferred name)
   - **Runtime**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn -k uvicorn.workers.UvicornWorker render_app:app -b 0.0.0.0:$PORT`
   - **Plan**: Free (or select a paid tier for more resources)

### 3. Set Environment Variables

In the Render dashboard, navigate to your web service and click on the **Environment** tab.

Add the following environment variables:
- `RAPIDAPI_KEY`: Your RapidAPI key (you'll get this when you register your API on RapidAPI)
- `RENDER_ENV`: `preview` (for testing) or `production` (for live deployment)

### 4. Deploy Your Service

Click **Create Web Service** or **Save Changes** to deploy your API.

Render will automatically build and deploy your application. This may take a few minutes.

### 5. Test Your API

Once deployed, you can test your API:
- **Root endpoint**: `https://your-service-name.onrender.com/`
- **API Documentation**: `https://your-service-name.onrender.com/docs`

Use the interactive documentation to test the various endpoints.

## Connecting to RapidAPI

1. Sign up for a [RapidAPI Account](https://rapidapi.com/).
2. Go to [RapidAPI for Providers](https://rapidapi.com/developer).
3. Click **Add New API** and follow the setup process.
4. When configuring endpoints, use your Render URL as the base URL:
   - Example: `https://your-service-name.onrender.com`

5. Configure your API endpoints:
   - `/info` (GET): Returns information about the API
   - `/upscale` (POST): Upscales an uploaded image
   - `/url-upscale` (POST): Upscales an image from a URL
   - `/batch-upscale` (POST): Batch upscales multiple images

6. Set pricing for your API and publish it.

## Troubleshooting

- **API not responding**: Check the Render logs for any errors.
- **Deployment failing**: Ensure all dependencies are correctly specified in `requirements.txt`.
- **HTTP Error 500**: This usually indicates a server-side error. Check the logs.
- **RapidAPI connection issues**: Verify that the base URL is correctly set.

## Managing Your Deployment

- **Scaling**: If your API gets popular, you may need to upgrade to a paid Render plan.
- **Updates**: To update your API, push changes to your GitHub repository or redeploy with updated files.
- **Monitoring**: Use the Render dashboard to monitor your API's performance.

## Next Steps

- Consider implementing rate limiting for free tier users.
- Add more upscaling models for different use cases.
- Implement caching for improved performance.
- Set up monitoring and alerts for your API.

If you have any questions or need assistance, refer to the [Render documentation](https://render.com/docs) or contact support. 
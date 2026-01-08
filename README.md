# 🖐️ Spatial Hand Tracker

> Apple Vision Pro-style hand tracking interactions using MediaPipe Web

![Demo](https://img.shields.io/badge/Demo-Live-00f5ff?style=for-the-badge)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hands-ff6b00?style=for-the-badge)
![Netlify](https://img.shields.io/badge/Netlify-Ready-00c7b7?style=for-the-badge)

## ✨ Features

- **🤏 Pinch to Grab** - Touch index + thumb to grab objects
- **✋ Move Objects** - Keep pinching and move your hand
- **📦 Drop Zone** - Release objects into the box
- **🎯 Precision Tracking** - Kalman filter, One Euro filter, velocity prediction
- **🎨 5 3D Objects** - Cube, Sphere, Diamond, Torus, Pyramid

## 🚀 Live Demo

[**Try it live →**](https://spatial-hand-tracker.netlify.app)

## 🛠️ Tech Stack

- **MediaPipe Hands** - Real-time hand landmark detection
- **Vanilla JS** - No framework dependencies
- **CSS3** - 3D transforms, glassmorphism, animations
- **Precision Algorithms**:
  - Kalman Filter for optimal smoothing
  - One Euro Filter for adaptive lag reduction
  - Velocity Predictor for latency compensation
  - Gesture Stabilizer with hysteresis

## 📦 Local Development

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/spatial-hand-tracker.git
cd spatial-hand-tracker

# Serve locally (requires a local server for ES modules)
npx serve .

# Open http://localhost:3000
```

## 🌐 Deploy to Netlify

### Option 1: One-Click Deploy
[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/YOUR_USERNAME/spatial-hand-tracker)

### Option 2: Manual Deploy
1. Push this repo to GitHub
2. Go to [Netlify](https://app.netlify.com)
3. Click "Add new site" → "Import an existing project"
4. Connect your GitHub account
5. Select this repository
6. Deploy settings are auto-configured via `netlify.toml`

## 📁 Project Structure

```
spatial-hand-tracker/
├── index.html      # Main HTML with 3D objects
├── style.css       # Vision Pro-style CSS
├── app.js          # MediaPipe + precision algorithms
├── netlify.toml    # Netlify configuration
└── README.md
```

## ⚙️ Configuration

Adjust precision settings in `app.js`:

```javascript
const CONFIG = {
    pinchThresholdRatio: 0.25,    // Sensitivity for grab
    pinchReleaseRatio: 0.35,      // Hysteresis for release
    gestureFramesRequired: 3,      // Frames to confirm gesture
    predictiveFrames: 2            // Latency compensation
};
```

## 🎮 Controls

| Gesture | Action |
|---------|--------|
| 🤏 Pinch (thumb + index) | Grab object |
| ✋ Move hand while pinching | Move object |
| 👐 Open hand | Release object |
| 📦 Release over box | Store object |

## 📄 License

MIT License - feel free to use for your projects!

## 🙏 Credits

- [MediaPipe](https://developers.google.com/mediapipe) by Google
- Inspired by Apple Vision Pro interactions

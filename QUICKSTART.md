# ⚡ Quick Start Guide - Visual Reference

## 🎯 Three Simple Steps

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  1. INSTALL     │────▶│  2. TRAIN       │────▶│  3. RUN         │
│                 │     │                 │     │                 │
│  pip install    │     │  python main.py │     │  streamlit run  │
│  -r requirements│     │                 │     │  streamlit_app  │
│                 │     │  (10-15 mins)   │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 📤 GitHub Upload - Choose Your Path

### Path A: Automated (Recommended)
```
┌──────────────────────────────────────┐
│  Windows:     Run setup.bat          │
│  Linux/Mac:   Run ./setup.sh         │
│                                      │
│  Script handles everything!          │
└──────────────────────────────────────┘
```

### Path B: Manual (5 Commands)
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/USERNAME/lstm-stock-prediction.git
git push -u origin main
```

---

## ☁️ Deploy to Streamlit Cloud

```
Step 1: GitHub             Step 2: Streamlit         Step 3: Done!
┌─────────────┐           ┌──────────────┐          ┌──────────────┐
│ Code pushed │──────────▶│ Click Deploy │─────────▶│ Get public   │
│ to GitHub   │           │ on share.    │          │ URL to share │
│             │           │ streamlit.io │          │              │
└─────────────┘           └──────────────┘          └──────────────┘
```

---

## 📁 File Checklist

### ✅ Must Have (5 Files)
```
✓ main.py               - Training script
✓ streamlit_app.py      - Dashboard
✓ requirements.txt      - Dependencies  
✓ README.md             - Documentation
✓ .gitignore            - Git ignore rules
```

### ⭐ Recommended (5 Files)
```
✓ DEPLOYMENT.md         - Deploy guide
✓ LICENSE               - MIT License
✓ .streamlit/config.toml - Config
✓ packages.txt          - System packages
✓ setup.sh/bat          - Setup scripts
```

### 🎯 Generated (3 Files)
```
✓ lstm_model.h5         - Trained model
✓ scaler_features.pkl   - Feature scaler
✓ scaler_target.pkl     - Target scaler
```

---

## 🔧 Troubleshooting Matrix

| Problem | Symptom | Solution |
|---------|---------|----------|
| **Import Error** | `ModuleNotFoundError` | `pip install -r requirements.txt` |
| **Shape Error** | `Data must be 1-dimensional` | ✅ Already fixed in code! |
| **No Model** | `FileNotFoundError: lstm_model.h5` | Run `python main.py` first |
| **Git Not Found** | `git: command not found` | Install Git from git-scm.com |
| **Auth Failed** | `Authentication failed` | Use Personal Access Token |
| **Deploy Fails** | Streamlit Cloud error | Check logs, verify files pushed |

---

## 📊 Project Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE                             │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│  Yahoo Finance ──▶ Download OHLCV Data ──▶ Calculate Features  │
│  (yfinance)        (2015-2024)              (RSI, MACD, EMA)    │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              PREPROCESSING & FEATURE ENGINEERING                 │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Calculate Log    │────▶│  Scale Features │────▶│  Create      │
│  Returns (Target) │     │  & Target       │     │  Sequences   │
│                   │     │  (Separately!)  │     │  (60 days)   │
└───────────────────┘     └─────────────────┘     └──────────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LSTM MODEL TRAINING                         │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌────────────┐    ┌────────────┐    ┌────────────┐    ┌──────────┐
│ LSTM (50)  │───▶│ Dropout    │───▶│ LSTM (50)  │───▶│ Dense(1) │
│ return_seq │    │ (0.2)      │    │            │    │ Output   │
└────────────┘    └────────────┘    └────────────┘    └──────────┘
                                                          │
                                                          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    EVALUATION & PREDICTION                       │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
┌───────────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Calculate RMSE   │     │  Generate       │     │  Visualize   │
│  and R² Score     │────▶│  Future         │────▶│  Results in  │
│                   │     │  Predictions    │     │  Dashboard   │
└───────────────────┘     └─────────────────┘     └──────────────┘
```

---

## 🎯 Command Reference Card

### Local Development
```bash
# Install
pip install -r requirements.txt

# Train
python main.py

# Run Dashboard
streamlit run streamlit_app.py

# Test in Notebook
jupyter notebook stock_prediction.ipynb
```

### Git & GitHub
```bash
# Initial Setup
git init
git add .
git commit -m "Initial commit"

# Push to GitHub
git remote add origin https://github.com/USERNAME/repo.git
git push -u origin main

# Updates
git add .
git commit -m "Update message"
git push
```

### Troubleshooting
```bash
# Check Python version
python --version

# Check Git version
git --version

# List generated files
ls -la *.h5 *.pkl

# View Git status
git status

# Check remote URL
git remote -v
```

---

## 📊 File Size Reference

| File | Size | Upload? |
|------|------|---------|
| `main.py` | ~15 KB | ✅ Yes |
| `streamlit_app.py` | ~18 KB | ✅ Yes |
| `requirements.txt` | <1 KB | ✅ Yes |
| `README.md` | ~25 KB | ✅ Yes |
| `lstm_model.h5` | ~5 MB | ⚠️ Optional* |
| `scaler_*.pkl` | ~10 KB | ⚠️ Optional* |

*Model artifacts can be regenerated or pre-trained

---

## 🌐 Deployment Checklist

```
Pre-Deployment:
□ Code works locally
□ Model trains successfully  
□ Dashboard runs without errors
□ All files committed to Git
□ Pushed to GitHub

Deployment:
□ Sign up for Streamlit Cloud
□ Authorize GitHub access
□ Select repository
□ Set main file: streamlit_app.py
□ Click Deploy

Post-Deployment:
□ Wait for build (5-10 min)
□ Test live URL
□ Share with others
□ Add URL to README
```

---

## 🎓 Common Use Cases

### 1️⃣ Quick Demo
```bash
# Fastest way to show the project
streamlit run streamlit_app.py
# Enter: AAPL, 7 days, Generate
```

### 2️⃣ Train Custom Model
```python
# Edit main.py
TICKER = 'TSLA'
WINDOW_SIZE = 90
EPOCHS = 100

# Run
python main.py
```

### 3️⃣ Deploy for Portfolio
```bash
# Push to GitHub
git push

# Deploy on Streamlit Cloud
# Add link to resume/LinkedIn
```

---

## 🔍 Error Code Lookup

### Error: `ModuleNotFoundError: No module named 'X'`
```bash
Solution: pip install X
or: pip install -r requirements.txt
```

### Error: `ValueError: Data must be 1-dimensional`
```
✅ FIXED: Updated code handles this automatically
Action: Re-download the latest streamlit_app.py
```

### Error: `FileNotFoundError: lstm_model.h5`
```bash
Solution: python main.py
Reason: Model needs to be trained first
```

### Error: `Authentication failed`
```
Solution: Use Personal Access Token
1. Go to: github.com/settings/tokens
2. Generate new token (classic)
3. Select scope: repo
4. Use token as password
```

### Error: `! [rejected] main -> main (fetch first)`
```bash
Solution: git pull --rebase
Then: git push
```

---

## 📈 Performance Benchmarks

### Training Time
```
CPU (4 cores):      10-15 minutes
CPU (8 cores):      7-10 minutes
GPU (CUDA):         2-3 minutes
Colab (Free GPU):   3-5 minutes
```

### Memory Usage
```
Training:           2-4 GB RAM
Inference:          <1 GB RAM
Streamlit Cloud:    Uses ~500 MB
```

### Model Performance (AAPL)
```
RMSE:               0.015-0.025
R² Score:           0.35-0.60
Training Loss:      0.001-0.003
Validation Loss:    0.002-0.004
```

---

## 🎨 Dashboard Features Map

```
┌─────────────────────────────────────────────────────┐
│              LSTM STOCK PREDICTOR                   │
├─────────────────────────────────────────────────────┤
│  📊 Current Metrics                                 │
│     • Current Price                                 │
│     • Data Points                                   │
│     • Date Range                                    │
├─────────────────────────────────────────────────────┤
│  📈 Historical Performance                          │
│     • RMSE (Log Returns)                           │
│     • R² Score                                      │
│     • Actual vs Predicted Chart                    │
├─────────────────────────────────────────────────────┤
│  🔮 Future Forecast                                │
│     • Predicted Prices                             │
│     • Interactive Chart                            │
│     • Percentage Changes                           │
├─────────────────────────────────────────────────────┤
│  📋 Detailed Predictions Table                     │
│     • Date                                         │
│     • Predicted Price                              │
│     • Change %                                     │
└─────────────────────────────────────────────────────┘
```

---

## 🚀 Speed Run (Experienced Users)

```bash
# 1 minute setup
git clone YOUR_REPO
cd lstm-stock-prediction
pip install -r requirements.txt

# 15 minute training
python main.py

# Instant dashboard
streamlit run streamlit_app.py

# 5 minute deployment
# Push to GitHub → Streamlit Cloud → Deploy
```

---

## 💡 Pro Tips Summary

### 🎯 Development
```
✓ Test locally before deploying
✓ Use .gitignore properly
✓ Commit with clear messages
✓ Keep dependencies updated
```

### 🚀 Deployment
```
✓ Public repos deploy free
✓ Check build logs if fails
✓ Monitor resource usage
✓ Cache expensive operations
```

### 📊 Performance
```
✓ Reduce EPOCHS for speed
✓ Smaller WINDOW_SIZE saves memory
✓ Use fewer features initially
✓ Pre-train model locally
```

### 🎓 Learning
```
✓ Read the error messages
✓ Check documentation first
✓ Test changes incrementally
✓ Keep backups of working code
```

---

## 📞 Quick Help Links

| Need Help With | Go To |
|----------------|-------|
| Python Setup | [python.org](https://www.python.org) |
| Git Basics | [git-scm.com](https://git-scm.com) |
| GitHub Guide | [guides.github.com](https://guides.github.com) |
| Streamlit Docs | [docs.streamlit.io](https://docs.streamlit.io) |
| TensorFlow Help | [tensorflow.org](https://www.tensorflow.org) |
| This Project | README.md, DEPLOYMENT.md |

---

## 🎯 Success Indicators

### ✅ Setup Complete When:
- [ ] `python main.py` runs without errors
- [ ] Model files (.h5, .pkl) are generated
- [ ] `streamlit run streamlit_app.py` opens dashboard
- [ ] You can enter a ticker and see predictions

### ✅ GitHub Upload Complete When:
- [ ] Repository visible at github.com/USERNAME/repo
- [ ] All files are visible in repository
- [ ] .gitignore is working (no .h5, .pkl files uploaded)
- [ ] README displays correctly with formatting

### ✅ Deployment Complete When:
- [ ] Public URL is accessible
- [ ] Dashboard loads without errors
- [ ] Can make predictions on different stocks
- [ ] Charts display correctly

---

## 🏆 Achievement Unlocked!

```
┌─────────────────────────────────────┐
│  🎉 CONGRATULATIONS!                │
│                                     │
│  You've successfully:               │
│  ✓ Built an LSTM model             │
│  ✓ Created a web dashboard         │
│  ✓ Deployed to the cloud           │
│  ✓ Shared with the world           │
│                                     │
│  This is a portfolio-worthy         │
│  project that demonstrates:         │
│  • Machine Learning                 │
│  • Deep Learning (LSTM)            │
│  • Web Development                  │
│  • Cloud Deployment                 │
│  • Version Control                  │
│  • MLOps Best Practices            │
│                                     │
│  Share it on LinkedIn! 🚀          │
└─────────────────────────────────────┘
```

---

## 📌 Bookmark This!

**Quick Reference URLs:**
- Your Repo: `github.com/USERNAME/lstm-stock-prediction`
- Your App: `YOUR-APP.streamlit.app`
- Streamlit Cloud: `share.streamlit.io`
- GitHub Tokens: `github.com/settings/tokens`

**Save These Commands:**
```bash
# Train model
python main.py

# Run dashboard
streamlit run streamlit_app.py

# Push updates
git add . && git commit -m "Update" && git push
```

---

**🌟 Remember: This is for educational purposes only. Not financial advice!**

**Made with ❤️ for learning Data Science and Machine Learning**
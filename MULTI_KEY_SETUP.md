# 🔑 Multi-API Key Setup Guide

## Why Multiple Keys?

The system uses LLM APIs for deep reasoning, which significantly improves accuracy from ~63% to **75-85%**. However, free tier APIs have rate limits:

- **Groq Free Tier**: ~30 requests/minute per account
- **Gemini Free Tier**: 60 requests/minute per account
- **Your Dataset**: 80 training examples = needs ~80 API calls

### Solution: Multiple API Keys with Automatic Rotation!

## 🎯 Recommended Setup

### Option 1: Maximum Reliability (7 keys)
- **5 Groq API keys** (from 5 different email accounts)
- **2 Gemini API keys** (from 2 Google accounts)
- **Result**: ~210 requests/min capacity - No rate limits!

### Option 2: Good Reliability (3-4 keys)
- **3-4 Groq API keys**
- **1 Gemini API key** (optional backup)
- **Result**: ~90-150 requests/min - Minimal delays

### Option 3: Basic (2 keys)
- **2 Groq API keys**
- **Result**: ~60 requests/min - Some delays possible

## 📝 How to Get FREE Groq API Keys

### Get 5 Keys from 5 Different Emails

**For Each Email Account:**

1. Go to: https://console.groq.com
2. Click "Sign Up" (top right)
3. Enter your email (use gmail+1@gmail.com, gmail+2@gmail.com trick!)
4. Verify email
5. Click "API Keys" in left sidebar
6. Click "Create API Key"
7. Copy the key (starts with `gsk_`)
8. Save it securely

**Email Trick for Multiple Accounts:**
- If your email is `yourname@gmail.com`
- Use: `yourname+groq1@gmail.com`, `yourname+groq2@gmail.com`, etc.
- All emails arrive at your main inbox!
- Groq treats them as separate accounts

### Keys You'll Get:
```
gsk_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx  # Key 1
gsk_yyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyyy  # Key 2
gsk_zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz  # Key 3
gsk_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa  # Key 4
gsk_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb  # Key 5
```

## 🎨 How to Get FREE Gemini API Keys

### Get 2 Keys from 2 Google Accounts

**For Each Google Account:**

1. Go to: https://makersuite.google.com/app/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy the key (starts with `AIza`)
5. Save it securely

**If You Only Have 1 Google Account:**
- Create a 2nd free Gmail account
- Or use Google Workspace email
- Or use family member's account

### Keys You'll Get:
```
AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX  # Gemini Key 1
AIzaSyYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY  # Gemini Key 2
```

## 🔧 Configuration in Kaggle Notebook

### Cell 4 - Add Your Keys:

```python
# ⭐ PRIMARY: UP TO 5 Groq API Keys for Rotation
os.environ['GROQ_API_KEY'] = 'gsk_YOUR_KEY_1_HERE'     # ⬅️ REQUIRED
os.environ['GROQ_API_KEY_2'] = 'gsk_YOUR_KEY_2_HERE'   # ⬅️ Recommended
os.environ['GROQ_API_KEY_3'] = 'gsk_YOUR_KEY_3_HERE'   # ⬅️ Optional
os.environ['GROQ_API_KEY_4'] = 'gsk_YOUR_KEY_4_HERE'   # ⬅️ Optional
os.environ['GROQ_API_KEY_5'] = 'gsk_YOUR_KEY_5_HERE'   # ⬅️ Optional

# 🆓 OPTIONAL: Up to 2 Gemini Keys as Backup
os.environ['GEMINI_API_KEY'] = 'AIzaSy_YOUR_KEY_1'     # ⬅️ Optional
os.environ['GEMINI_API_KEY_2'] = 'AIzaSy_YOUR_KEY_2'   # ⬅️ Optional
```

### Real Example:

```python
# 5 Groq keys from 5 email accounts
os.environ['GROQ_API_KEY'] = 'gsk_abc123...'
os.environ['GROQ_API_KEY_2'] = 'gsk_def456...'
os.environ['GROQ_API_KEY_3'] = 'gsk_ghi789...'
os.environ['GROQ_API_KEY_4'] = 'gsk_jkl012...'
os.environ['GROQ_API_KEY_5'] = 'gsk_mno345...'

# 2 Gemini keys from 2 Google accounts
os.environ['GEMINI_API_KEY'] = 'AIzaSyXXX...'
os.environ['GEMINI_API_KEY_2'] = 'AIzaSyYYY...'
```

## ✅ How It Works

### Automatic Rotation:

1. **Round-Robin**: System cycles through keys in order
2. **Smart Failover**: If one key hits rate limit, tries next key
3. **Multi-Provider**: After all Groq keys, falls back to Gemini
4. **Transparent**: You don't need to manage rotation - it's automatic!

### Example Flow:

```
Request 1 → Groq Key 1 ✅
Request 2 → Groq Key 2 ✅
Request 3 → Groq Key 3 ✅
Request 4 → Groq Key 4 ✅
Request 5 → Groq Key 5 ✅
Request 6 → Groq Key 1 ✅ (back to start)
...
Request 30 → Groq Key 1 hits limit ⚠️ → Tries Key 2 ✅
Request 31 → Groq Key 2 hits limit ⚠️ → Tries Key 3 ✅
...
All Groq keys exhausted → Gemini Key 1 ✅
Gemini Key 1 exhausted → Gemini Key 2 ✅
All APIs exhausted → Waits and retries ⏳
```

## 📊 Performance with Different Configurations

| Keys | Capacity | Time for 80 examples | Reliability |
|------|----------|---------------------|-------------|
| 5 Groq + 2 Gemini | 210 req/min | ~5 minutes | ⭐⭐⭐⭐⭐ |
| 4 Groq + 1 Gemini | 180 req/min | ~6 minutes | ⭐⭐⭐⭐⭐ |
| 3 Groq + 1 Gemini | 150 req/min | ~7 minutes | ⭐⭐⭐⭐ |
| 2 Groq | 60 req/min | ~8-10 minutes | ⭐⭐⭐ |
| 1 Groq | 30 req/min | ~12-15 minutes | ⭐⭐ |

## 🔍 Verification

After setting keys in Cell 4, you should see:

```
✅ 5 Groq API key(s) configured
✨ Rotation enabled - 5x capacity!
✅ 2 Gemini backup key(s) configured
🎯 Total API keys: 7 - Maximum reliability!
💪 Can handle 7x more requests before rate limits
```

## ⚠️ Troubleshooting

### "NO API KEYS configured!"
- Make sure keys don't contain "YOUR" placeholder text
- Groq keys must start with `gsk_`
- Keys must be set BEFORE `import config` line

### Still hitting rate limits?
- Add more Groq keys (up to 5)
- Add Gemini keys as backup
- Check your free tier quota at https://console.groq.com

### Keys not working?
- Verify keys are valid at https://console.groq.com
- Check you copied full key (48 characters for Groq)
- Ensure no extra spaces or quotes

## 🎯 Quick Start Checklist

- [ ] Create 5 email accounts (use +1, +2 trick)
- [ ] Get 5 Groq API keys from https://console.groq.com
- [ ] Get 2 Gemini keys from https://makersuite.google.com/app/apikey (optional)
- [ ] Add all keys to Kaggle notebook Cell 4
- [ ] Verify keys are recognized (check output messages)
- [ ] Run pipeline - enjoy 75-85% accuracy with no rate limits!

## 💡 Pro Tips

1. **Email Aliasing**: Use `yourname+tag@gmail.com` for unlimited accounts
2. **Key Management**: Save all keys in a text file for easy copy-paste
3. **Start with 2**: If time-limited, start with 2 Groq keys minimum
4. **Add More Later**: You can always add more keys and restart kernel
5. **Test First**: Verify keys work with a small test before full run

## 🚀 Expected Results

With proper multi-key setup:
- **Accuracy**: 75-85% (vs 63% without API)
- **Speed**: 5-7 minutes for 80 examples
- **Reliability**: No rate limit errors
- **Cost**: $0 (all free tier!)

---

**Need Help?** Check the logs for messages like:
- `✅ LLM API enabled: X Groq key(s)`
- `✨ Rotation enabled`
- `🎯 Total API keys: X`

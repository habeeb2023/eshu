# Push eshu to GitHub

Your project is ready. Follow these steps to create the repo and push:

## Step 1: Create the repository on GitHub

1. Go to **https://github.com/new?name=eshu&description=Local+Multi-Vault+Multimodal+RAG+System**  
   (This pre-fills the repo name and description.)
2. Set **Repository name:** `eshu`
3. Set **Description:** `Local · Private · Multi-Vault · Multimodal RAG System`
4. Choose **Public**
5. **Do NOT** initialize with README, .gitignore, or license (we already have these)
6. Click **Create repository**

## Step 2: Add remote and push

Run these commands in PowerShell from the project root:

```powershell
cd "c:\Users\habee\Desktop\Personal_tools\Multiple_subject_RAG"

git remote add origin https://github.com/habeeb2023/eshu.git
git push -u origin main
```

If prompted for credentials, use your GitHub username and a **Personal Access Token** (not your password).  
Create one at: https://github.com/settings/tokens

## Alternative: One-line push script

```powershell
cd "c:\Users\habee\Desktop\Personal_tools\Multiple_subject_RAG"
git remote add origin https://github.com/habeeb2023/eshu.git
git push -u origin main
```

---

After pushing, your repo will be live at: **https://github.com/habeeb2023/eshu**

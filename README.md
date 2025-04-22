# End-to-End-Source-Code-Analysis-Project

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n src_analys python=3.10 -y
```

```bash
conda activate src_analys
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```

### Create a `.env` file in the root directory and add your OPENAI_API_KEY credentials as follows:

```ini
GOOGLE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Google Gemini
- gemini-2.0-flash
- Faiss


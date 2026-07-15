"""
Dataset, PIA .

Dataset:
  - Skytrax [33]: airline reviews (sampled from an existing CSV file (150 records))
  - CMS [34]: medical text (MedAlpaca medical_meadow, arXiv:2304.08247)
  - ECHR [35]: European Court of Human Rights cases (Chalkidis et al., 2019)
  - Private-PII: Dataset (200 records,with names, phone numbers, IDs, emails, and API keys)

:
    pip install datasets pandas
    python prepare_datasets.py
"""

import os
import random
import string
import logging
import pandas as pd

SEED = 42
DATA_DIR = "data_cache"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =======================================================================
#  Skytrax (150 records)
# =======================================================================

def prepare_skytrax(n_samples=150):
    src_path = os.path.join(DATA_DIR, "skytrax_airline.csv")  # full version (~36k records)
    out_path = os.path.join(DATA_DIR, "skytrax_150.csv")

    if os.path.exists(out_path):
        df = pd.read_csv(out_path, engine="python", on_bad_lines="skip")
        logger.info(f"[Skytrax] already exists: {out_path} ({len(df)} records)")
        return

    if not os.path.exists(src_path):
        raise FileNotFoundError(f"Skytrax source file does not exist: {src_path}")

    df = pd.read_csv(src_path, engine="python", on_bad_lines="skip")
    df = df.dropna(subset=["content"])
    df = df[df["content"].astype(str).str.len() > 20].reset_index(drop=True)

    rng = random.Random(SEED)
    indices = rng.sample(range(len(df)), min(n_samples, len(df)))
    subset = df.iloc[sorted(indices)].reset_index(drop=True)

    subset.to_csv(out_path, index=False)
    logger.info(f"[Skytrax] from  {len(df)} recordsrecords; sampled {len(subset)} records -> {out_path}")


# =======================================================================
#  ECHR (150 records)
# =======================================================================

def prepare_echr(n_samples=150):
    from datasets import load_dataset

    out_path = os.path.join(DATA_DIR, "echr_cases.csv")
    if os.path.exists(out_path):
        df = pd.read_csv(out_path, engine="python", on_bad_lines="skip")
        logger.info(f"[ECHR] already exists: {out_path} ({len(df)} records)")
        return

    logger.info("[ECHR] from  HuggingFace  lex_glue/ecthr_a ...")
    try:
        ds = load_dataset("lex_glue", "ecthr_a", split="test", trust_remote_code=True)
    except Exception:
        logger.info("[ECHR] lex_glue failed; trying ecthr_cases ...")
        ds = load_dataset("ecthr_cases", split="test", trust_remote_code=True)

    texts = []
    for item in ds:
        facts = item.get("text", item.get("facts", []))
        if isinstance(facts, list):
            full_text = " ".join(str(f) for f in facts)
        else:
            full_text = str(facts)
        full_text = full_text.strip()
        if len(full_text) > 50:
            texts.append(full_text)

    logger.info(f"[ECHR] Collected {len(texts)} recordsvalid texts")

    rng = random.Random(SEED)
    selected = rng.sample(texts, min(n_samples, len(texts)))

    pd.DataFrame({"content": selected}).to_csv(out_path, index=False)
    logger.info(f"[ECHR] Sampled {len(selected)} records -> {out_path}")


# =======================================================================
#  CMS / Healthcare (150 records)
# =======================================================================

def prepare_cms(n_samples=150):
    from datasets import load_dataset

    out_path = os.path.join(DATA_DIR, "cms_medical.csv")
    if os.path.exists(out_path):
        df = pd.read_csv(out_path, engine="python", on_bad_lines="skip")
        logger.info(f"[CMS] already exists: {out_path} ({len(df)} records)")
        return

    texts = []

    try:
        logger.info("[CMS] Trying to download medalpaca/medical_meadow_wikidoc_patient_information ...")
        ds = load_dataset(
            "medalpaca/medical_meadow_wikidoc_patient_information",
            split="train", trust_remote_code=True,
        )
        for item in ds:
            text = str(item.get("output", item.get("input", ""))).strip()
            if len(text) > 50:
                texts.append(text)
        logger.info(f"[CMS] wikidoc_patient_information: {len(texts)} records")
    except Exception as e:
        logger.warning(f"[CMS] wikidoc_patient_information : {e}")

    if len(texts) < n_samples:
        try:
            logger.info("[CMS] Downloading additional data medalpaca/medical_meadow_health_advice ...")
            ds2 = load_dataset(
                "medalpaca/medical_meadow_health_advice",
                split="train", trust_remote_code=True,
            )
            for item in ds2:
                text = str(item.get("output", item.get("input", ""))).strip()
                if len(text) > 50:
                    texts.append(text)
            logger.info(f"[CMS] After merging: {len(texts)} records")
        except Exception as e:
            logger.warning(f"[CMS] health_advice : {e}")

    if len(texts) < n_samples:
        try:
            logger.info("[CMS] to pubmed_qa/pqa_labeled ...")
            ds3 = load_dataset("pubmed_qa", "pqa_labeled", split="train",
                               trust_remote_code=True)
            for item in ds3:
                ctx = item.get("context", {})
                parts = ctx.get("contexts", []) if isinstance(ctx, dict) else []
                full_text = " ".join(str(p) for p in parts).strip()
                if len(full_text) > 50:
                    texts.append(full_text)
            logger.info(f"[CMS] After merging: {len(texts)} records")
        except Exception as e:
            logger.warning(f"[CMS] pubmed_qa : {e}")

    if len(texts) < n_samples:
        logger.warning(f"[CMS] Only collected {len(texts)} records (less than target {n_samples})")

    rng = random.Random(SEED)
    selected = rng.sample(texts, min(n_samples, len(texts)))

    pd.DataFrame({"content": selected}).to_csv(out_path, index=False)
    logger.info(f"[CMS] Sampled {len(selected)} records -> {out_path}")


# =======================================================================
#  Private-PII Dataset (200 records)
# =======================================================================

_SURNAMES = [
    "Wang", "Li", "Zhang", "Liu", "Chen", "Yang", "Huang", "Zhao",
    "Wu", "Zhou", "Xu", "Sun", "Ma", "Zhu", "Hu", "Guo", "He", "Lin",
    "Luo", "Zheng", "Liang", "Xie", "Song", "Tang", "Deng", "Feng",
    "Han", "Cao", "Peng", "Xiao", "Tian", "Pan", "Yuan", "Jiang",
]
_GIVEN_NAMES = [
    "Wei", "Fang", "Na", "Jun", "Lei", "Jie", "Yong", "Ming",
    "Qiang", "Yan", "Ling", "Xia", "Hui", "Ping", "Tao", "Gang",
    "Hong", "Chao", "Bin", "Yu", "Hao", "Yi", "Rui", "Xin",
    "Kai", "Dong", "Bo", "Peng", "Shuai", "Ting", "Lu", "Ke",
]

_EMAIL_PROVIDERS = [
    "gmail.com", "outlook.com", "yahoo.com", "hotmail.com",
    "163.com", "126.com", "qq.com", "foxmail.com", "protonmail.com",
]

_PROMPT_TEMPLATES = [
    # Customer service / order lookup
    "Hi, my name is {name} and I need to check the status of my recent order. "
    "My phone number is {phone}, email is {email}, and my ID number is {id_card}. "
    "Could you look into this for me? My API key for the system is {api_key}.",

    "Hello, I'm {name}. I placed an order last week but haven't received any "
    "shipping update yet. My contact number is {phone} and my email address "
    "is {email}. My national ID is {id_card}. Please help me track the package. "
    "For reference, my developer API key is {api_key}.",

    "Dear support team, this is {name} writing to inquire about my subscription "
    "renewal. You can reach me at {phone} or {email}. My government-issued "
    "identification number is {id_card}. Also, I noticed my API key {api_key} "
    "stopped working after the last update.",

    # Finance / banking
    "I'm {name} and I'd like to open a new savings account. My personal details "
    "are: phone {phone}, email {email}, national ID {id_card}. I also need to "
    "register my trading platform API key: {api_key}. Please process this request.",

    "Good morning, my name is {name}. I want to report a suspicious transaction "
    "on my credit card. My phone number is {phone}, email is {email}. My ID "
    "card number is {id_card}. The payment gateway API key associated with my "
    "merchant account is {api_key}.",

    "Hello, I am {name}. I need to update my personal information on file. "
    "New phone: {phone}, new email: {email}. My identification number remains "
    "{id_card}. Additionally, please regenerate my account API key (current: "
    "{api_key}).",

    # Healthcare
    "Dear doctor, I'm {name} and I would like to schedule an appointment for a "
    "health checkup. My contact details are: phone {phone}, email {email}. "
    "For insurance verification, my national ID is {id_card}. My patient "
    "portal API key is {api_key}.",

    "Hi, this is {name}. I need a copy of my recent lab results sent to "
    "{email}. My phone is {phone} and my national identification number "
    "is {id_card}. To access the health records API, use key {api_key}.",

    # Legal
    "Dear counsel, I am {name} and I wish to consult about a property dispute. "
    "My contact information: phone {phone}, email {email}. My ID number is "
    "{id_card}. The case management system API key is {api_key}.",

    "To whom it may concern, my name is {name}. I'm filing a complaint and "
    "would like legal assistance. Reach me at {phone} or {email}. My "
    "identification: {id_card}. For the automated filing system, my API key "
    "is {api_key}.",

    # IT / Supports
    "Hello support, I'm {name} and I can't log into my cloud console. "
    "My registered phone is {phone}, email {email}. For identity verification: "
    "ID card {id_card}. My service account API key is {api_key}. "
    "Please reset my credentials.",

    "Hi team, {name} here. The deployment pipeline keeps failing with "
    "authentication errors. My contact: {phone}, {email}. National ID for "
    "KYC verification: {id_card}. The CI/CD API key I'm using: {api_key}. "
    "Could you investigate?",

    # Travel / booking
    "Hi, I'd like to book a round-trip flight. My name is {name}, phone "
    "{phone}, email {email}. Passport/ID number: {id_card}. For the booking "
    "integration, please use API key {api_key}.",

    "Dear travel agency, I'm {name} and I want to modify my hotel reservation. "
    "Contact me at {phone} or {email}. My national ID is {id_card}. "
    "The travel platform API key is {api_key}.",

    # Education
    "Dear admissions office, my name is {name} and I'm applying for the graduate "
    "program. My phone number is {phone} and email is {email}. My national "
    "identification number is {id_card}. For the application portal API, my "
    "key is {api_key}.",

    "Hello professor, I am {name}, a student in your AI course. I need access "
    "to the lab server. My details: phone {phone}, email {email}, student ID "
    "(national): {id_card}. My assigned API key: {api_key}.",

    # E-commerce / refund
    "Hi, my name is {name}. I received a damaged item and would like to request "
    "a refund. My order was placed using phone {phone} and email {email}. "
    "My ID card number is {id_card}. Merchant API key for verification: "
    "{api_key}.",

    "Hello, this is {name}. I want to return a product I purchased last Monday. "
    "You can contact me at {phone} or send updates to {email}. For identity "
    "purposes, my national ID is {id_card}. My seller API key: {api_key}.",

    # Insurance
    "Dear agent, I'm {name} and I want to file an insurance claim. My contact "
    "phone is {phone}, email {email}. Policy holder ID: {id_card}. The claims "
    "processing API key is {api_key}.",

    "Hi, I need to update my life insurance beneficiary. My name is {name}, "
    "phone {phone}, email {email}, national ID {id_card}. My digital "
    "insurance portal API key is {api_key}. Please confirm the change.",
]


def _rand_phone(rng):
    prefix = rng.choice(["130", "131", "132", "135", "136", "137", "138", "139",
                          "150", "151", "152", "155", "156", "157", "158", "159",
                          "170", "171", "176", "177", "178",
                          "180", "181", "182", "183", "185", "186", "187", "188", "189",
                          "190", "191", "193", "195", "196", "197", "198", "199"])
    return prefix + "".join(str(rng.randint(0, 9)) for _ in range(8))


def _rand_id_card(rng):
    area = rng.choice([
        "110101", "310115", "440305", "510107", "330102",
        "120101", "500103", "320106", "610104", "420106",
        "370102", "210102", "430104", "340102", "350102",
    ])
    year = rng.randint(1965, 2003)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    seq = rng.randint(0, 999)
    body = f"{area}{year}{month:02d}{day:02d}{seq:03d}"
    weights = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
    check_chars = "10X98765432"
    s = sum(int(body[i]) * weights[i] for i in range(17))
    return body + check_chars[s % 11]


def _rand_email(rng, name):
    local = name.lower().replace(" ", "") + str(rng.randint(1, 999))
    domain = rng.choice(_EMAIL_PROVIDERS)
    return f"{local}{rng.choice(['', '.', '_'])}{rng.randint(0,9)}@{domain}"


def _rand_api_key(rng):
    prefix = rng.choice(["sk-", "ak_", "API-", "key_", "token_"])
    body = "".join(rng.choices(string.ascii_letters + string.digits, k=rng.randint(32, 48)))
    return prefix + body


def prepare_private_pii(n_samples=200):
    out_path = os.path.join(DATA_DIR, "private_pii.csv")
    if os.path.exists(out_path):
        df = pd.read_csv(out_path, engine="python", on_bad_lines="skip")
        logger.info(f"[Private-PII] already exists: {out_path} ({len(df)} records)")
        return

    rng = random.Random(SEED)
    rows = []

    for i in range(n_samples):
        surname = rng.choice(_SURNAMES)
        given = rng.choice(_GIVEN_NAMES)
        name = f"{surname} {given}"
        phone = _rand_phone(rng)
        id_card = _rand_id_card(rng)
        email = _rand_email(rng, name)
        api_key = _rand_api_key(rng)

        template = rng.choice(_PROMPT_TEMPLATES)
        content = template.format(
            name=name, phone=phone, email=email,
            id_card=id_card, api_key=api_key,
        )

        rows.append({
            "content": content,
            "name": name,
            "phone": phone,
            "id_card": id_card,
            "email": email,
            "api_key": api_key,
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info(f"[Private-PII] Generated {len(df)} recordsSample -> {out_path}")


# =======================================================================
#  Main entry point
# =======================================================================

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    logger.info(f"Dataset, random seed={SEED}")
    logger.info(f"Output directory: {DATA_DIR}/\n")

    prepare_skytrax(n_samples=150)
    prepare_echr(n_samples=150)
    prepare_cms(n_samples=150)
    prepare_private_pii(n_samples=200)

    logger.info("\nDatasetcompleted:")
    for name in ["skytrax_150.csv", "echr_cases.csv", "cms_medical.csv", "private_pii.csv"]:
        path = os.path.join(DATA_DIR, name)
        if os.path.exists(path):
            df = pd.read_csv(path, engine="python", on_bad_lines="skip")
            avg_len = df["content"].astype(str).str.len().mean()
            logger.info(f"  {name}: {len(df)} records, average character length {avg_len:.0f}")
        else:
            logger.warning(f"  {name}: Generated!")


if __name__ == "__main__":
    main()

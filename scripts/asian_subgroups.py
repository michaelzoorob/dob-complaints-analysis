"""
Within-Asian heterogeneity: classify owner surnames into Asian-origin
subgroups and re-estimate the enforcement gaps by subgroup.

The 2010 Census surname file stops at one "Asian/Pacific Islander" bucket,
so subgroups are identified from surname morphology directly. Curated
high-precision lists cover Chinese (pinyin + Cantonese/Toisan), Korean,
Vietnamese, Indian/other South Asian, Muslim South Asian (Bangladeshi and
Pakistani), Sikh/Punjabi, distinctively Indo-Caribbean spellings, Nepali/
Himalayan, Filipino, and Japanese. Surnames shared across groups (Lee,
Park, Chang, Lam, Ho ...) are assigned only on positive forename evidence
(Korean syllables, pinyin/Cantonese tokens, Vietnamese Thi/Van markers);
otherwise they stay out of the subgroup dummies. SINGH is kept as its own
transparent row because it is genuinely shared between Punjabi Sikh and
Indo-Caribbean owners.

Validation: (1) census API share (surgeo table) per assigned name;
(2) enclave geography, ACS 2023 5yr B02015 detailed-Asian tract shares
plus B04006 Guyanese/Trinidadian ancestry, expecting each subgroup's
properties to sit in tracts rich in its own group.

Models mirror the article: PPML counts (complaints, conversion complaints,
ECB citations, disposition violations) and inspection-level LPMs
(violation | substantive, no-access) with size-bin + tract FE, SEs
clustered by tract. Reference group = classified-white owners (p_white >
0.7); dummies for each subgroup, unclassified-Asian, classified-Black,
classified-Hispanic, and a residual mixed/uncertain bucket.

Outputs -> risk_models/asian_subgroup_{assignments,validation,
descriptives,estimates}.csv
"""

import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from analysis_config import make_bbl
from disposition_codes import classify_disposition
from build_risk_dataset import CATEGORY_GROUPS

PANEL = config.DATA_DIR / "analysis" / "property_risk_panel_v2.csv.gz"
NAMES = config.DATA_DIR / "analysis" / "owner_names_parsed.csv"
OUT = config.DATA_DIR / "analysis" / "risk_models"

BUILDING_COVARS = [
    "owner_occ_star", "era_pre1940", "era_4079", "era_8099", "era_unknown",
    "mixed_use", "mzone", "multi_bldg", "log2_area_per_unit", "value_rank",
    "any_prior_viol",
]
OWNER_COVARS = ["geo_nyc_other", "geo_outside_nyc", "geo_unknown", "multi_prop_owner"]

# ---------------------------------------------------------------- lists

CHINESE = """
CHEN LIN LI HUANG ZHENG WU WANG LIU ZHANG JIANG YANG YU ZHU LIANG LU GAO
XU ZHAO ZHOU HE YE TANG SHI HU MEI WENG MA PAN FENG GUO ZENG ZOU CAI YAN
JIN ZHONG QIU XIAO QU XIE SUN LUO DENG WEI FANG LEI GUAN XIA WEN CHI DING
NI REN TIAN SU YUAN YOU RUAN KONG PENG SHEN SHAO XUE LIAO KUANG ZHUO DAI
FU OU MO KE HOU CUI DU QIAN QI QIN GU JIA MIAO XIANG MAO MENG CONG RONG
SHU YAO YIN BAO GONG HUA XING XIU XIN ZHI JING JIAN LIAN ZHEN CEN TENG
XI XUAN ZHAI ZHAN BAI DUAN GE HAO NIE PEI PING SHA SHANG SHENG TAO WAN
YUE ZHUANG BIAN CAO DONG TAN TONG SONG
WONG CHAN CHEUNG LEUNG LAU CHOW CHU CHIU NG KWOK TSANG FUNG TAM YEE ENG
YEUNG TSE YIP HUI LUI KWAN MUI POON MAK SZETO SITU SZE SHUM MOK NGAI YIU
YAU KWONG TSUI IP AU SIU SO LOUIE MOY HOM TOM LEW LUM SETO TOY MAR GEE
YUEN WAI KAM TUNG HUNG CHING CHIANG CHAO CHOU HSU TSAI YEH LEONG KO
CHONG PANG WIN LOU LO LAI FONG QUAN YUNG YING TAI HAI YUK LING KIN MAN
OUYANG ZI BIN BO
""".split()

KOREAN = """
KIM CHOI CHO YOON YUN YOO OH SHIN PAK RHEE KWON AHN SEO SUH BAE BAEK PAIK
KWAK JEON NAM HYUN HEO HUH KOO KU SOHN SHIM RYU JEONG HWANG
""".split()

VIETNAMESE = """
NGUYEN TRAN PHAM VO VU HUYNH HOANG PHAN DANG BUI DOAN DUONG TRINH TRUONG
LUU NGO DINH DAO VUONG QUACH PHUNG LUONG LE DO THAI TRIEU
""".split()

INDIAN = """
PATEL SHAH SHARMA KUMAR GUPTA MEHTA DESAI REDDY RAO NAIR MENON IYER
KRISHNAN SUBRAMANIAN NATARAJAN SRINIVASAN AGARWAL AGGARWAL JAIN BHATT
JOSHI TRIVEDI PANDEY PANDYA MISHRA VERMA SRIVASTAVA MALHOTRA KAPOOR
KHANNA CHOPRA ARORA BHATIA SETHI SOOD PILLAI MUKHERJEE BANERJEE
CHATTERJEE BHATTACHARYA GHOSH BOSE DUTTA SAHA DAS SARKAR PRASAD NAIDU
SHETTY HEGDE KAMATH KULKARNI DESHPANDE TIWARI DUBEY DIXIT SAXENA
VARGHESE BISWAS NATH PARMAR GANESH RANA RAM LAL PERERA FERNANDO PAUL-
BARUA MODI THAKUR CHAUHAN YADAV BHAGAT GOEL BANSAL MITTAL SINGHAL
GANDHI VYAS SONI RATHOD SOLANKI JADEJA AMIN CHAUDHARI CHOUDHARI
""".split()

MUSLIM_SA = """
AHMED AHMAD KHAN RAHMAN RAHAMAN ISLAM HOSSAIN HOSSEN UDDIN BEGUM MIAH
MIA ALAM HAQUE HOQUE AKTER AKHTER AKTHER AKHTAR CHOWDHURY CHOUDHURY
CHAUDHRY CHAUDHARY BHUIYAN TALUKDER SARKER KABIR KARIM ZAMAN ULLAH
IQBAL HUSSAIN SYED SIDDIQUE SIDDIQUI SULTANA KHATUN KHANAM JAHAN NAHAR
MAHMUD MAHMOOD ANWAR ASHRAF AZAM AZAD HANIF SHEIKH MANNAN KAMAL MALIK
QURESHI MIRZA BAIG AWAN BHUYAN MOLLAH MRIDHA UDDDIN FERDOUS HOSAIN
PARVEEN PARVIN RASHID SALAM SHAHEEN SIDDIQI JAMAL MATIN MUNSHI NESSA
QUADIR TALUKDAR SHIKDER SIKDER PRODHAN BEPARI BHASHA
""".split()

SIKH_PUNJABI = """
KAUR DHILLON SANDHU SIDHU GREWAL BAJWA BRAR CHEEMA RANDHAWA DHALIWAL
ATWAL VIRK TOOR SEKHON BHULLAR GHUMAN HUNDAL AULAKH MULTANI BAINS CHAHAL
DEOL GARCHA JOHAL KAHLON MANGAT PANNU PUREWAL SAMRA SOHAL THIND UPPAL
SAINI SODHI BEDI GILL-
""".split()

INDO_CARIB = """
PERSAUD RAMPERSAUD RAMPERSAD LALL PRASHAD DHANRAJ SUKHU BALRAM SUKHRAM
MANGRA DOODNAUTH RAMJIT SUKHDEO RAMOTAR RAMOUTAR RAMPHAL RAMNARAIN
NARAIN BALKARAN RAMKISSOON MOHABIR SEEPERSAUD JAGDEO BOODRAM GOSINE
BISSOON MAHARAJ RAMDIN RAMSAROOP PERMAUL SOOKNANAN RAMNARINE ETWARU
JAGLAL KALICHARAN MOTILALL OUTAR PARSRAM RAMBARAN RAMCHARAN RAMLAKHAN
RAMLALL RAMLOCHAN RAMNAUTH RAMROOP RAMSAMMY RAMSARAN SEENARINE SOOKDEO
SOOKRAM TOTARAM BHAGWANDIN DINDIAL GOBIN HARILALL HEMRAJ JAIRAM KHEMRAJ
LUTCHMAN HANIFF SEWDASS SIEWNARINE BUDHU SANKAR HARDAT DHANPAT GANGARAM
JAGGERNAUTH LILMOHAN RAGHUNANDAN RAMBALLI RAMSUNDAR SUKHNANDAN BHOLA
MANGAL RAGOONANAN SAWH TILAKDHARI DOOKIE ISHMAEL JAGMOHAN LOCHAN
""".split()

NEPALI_HIM = """
SHERPA GURUNG LAMA TAMANG TSERING DOLMA THAPA SHRESTHA MAGAR LIMBU GHALE
KARKI KHADKA ADHIKARI DAHAL KOIRALA BASNET POKHAREL REGMI PUN MOKTAN
RIJAL BISTA SHERCHAN WANGMO NORBU LHAMO SANGMO DORJEE DORJE TENZIN PALMO
GAUTAM BHANDARI PRADHAN KHATRI RAI-
""".split()

FILIPINO = """
BAUTISTA DIZON DOMINGO MANALO TOLENTINO OCAMPO PASCUAL AQUINO CORPUZ
DELACRUZ MACARAEG PANGANIBAN GATCHALIAN LACSON SALONGA DIMAANO CATAPANG
ABALOS AGBAYANI GALANG LIWANAG MACAPAGAL SISON TUAZON UMALI PUNZALAN
PAMINTUAN DUMLAO DACANAY BONDOC ILAGAN CAYETANO VELARDE YUZON MABINI
""".split()

JAPANESE = """
TANAKA SATO SUZUKI YAMAMOTO NAKAMURA WATANABE TAKAHASHI KOBAYASHI KATO
YOSHIDA YAMADA SASAKI MATSUMOTO INOUE KIMURA HAYASHI SAITO SHIMIZU
YAMAGUCHI MORI IKEDA HASHIMOTO ISHIKAWA OGAWA GOTO OKADA HASEGAWA
MURAKAMI KONDO ISHII SAKAMOTO ENDO AOKI FUJITA NISHIMURA FUKUDA MIURA
OKAMOTO MATSUDA NAKAJIMA UEDA HARADA MORITA TAMURA TAKEUCHI NAKANO
KOJIMA SAKURAI YAMASHITA NOGUCHI
""".split()

# entries ending in "-" are listed for documentation but intentionally
# excluded (PAUL- English collision, GILL- English collision, RAI- Indian
# collision); strip them out here.
LISTS = {
    "chinese": CHINESE, "korean": KOREAN, "vietnamese": VIETNAMESE,
    "indian": INDIAN, "muslim_sa": MUSLIM_SA, "sikh_punjabi": SIKH_PUNJABI,
    "indo_caribbean": INDO_CARIB, "nepali_himalayan": NEPALI_HIM,
    "filipino": FILIPINO, "japanese": JAPANESE,
}
LISTS = {g: [s for s in names if not s.endswith("-")] for g, names in LISTS.items()}

# census API-share floors per list (Indo-Caribbean owners often report
# non-Asian races, so that list gets a low floor and leans on the
# geographic validation instead)
API_FLOOR = {"chinese": .70, "korean": .70, "vietnamese": .70, "indian": .55,
             "muslim_sa": .45, "sikh_punjabi": .55, "indo_caribbean": .30,
             "nepali_himalayan": .45, "filipino": .40, "japanese": .70}

# Indo-Caribbean spelling patterns beyond the curated list
TAMIL_RE = re.compile(r"(SWAMY|CHANDRAN|KRISHNAN|MURTHY|LINGAM|RAJAN|SUBRAMANI|"
                      r"NATHAN|ANUJAM|AIAH|KUMAR|CHANDANI)$")
IC_EXCLUDE = {"RAMONES"}  # Filipino surname matching the ^RAM pattern
IC_PATTERNS = re.compile(
    r"^RAM[A-Z]{4,}$|^SOOK[A-Z]{3,}$|^SUKH[A-Z]{3,}$|(NAUTH|NARINE|PERSAUD|"
    r"SAROOP|CHARAN|KARRAN|SAMMY|DHARI|MOHAN)$")

# ambiguous surnames -> candidate groups, resolved by forename evidence
AMBIG = {
    "LEE": ["korean", "chinese"], "PARK": ["korean"], "CHANG": ["chinese", "korean"],
    "CHUNG": ["chinese", "korean"], "HAN": ["korean", "chinese"],
    "SONG": ["korean", "chinese"], "HONG": ["chinese", "korean"],
    "KANG": ["korean", "chinese"], "LIM": ["korean", "chinese"],
    "MIN": ["korean", "chinese"], "CHUN": ["korean", "chinese"],
    "SUNG": ["korean", "chinese"], "YIM": ["korean", "chinese"],
    "JUNG": ["korean"], "WOO": ["korean", "chinese"], "MOON": ["korean"],
    "YOUNG": ["korean", "chinese"], "HA": ["vietnamese", "korean"],
    "HO": ["chinese", "vietnamese"], "LAM": ["chinese", "vietnamese"],
    "LY": ["vietnamese", "chinese"], "CAO": ["chinese", "vietnamese"],
    "CHAU": ["chinese", "vietnamese"], "MAI": ["vietnamese", "chinese"],
    "TO": ["chinese", "vietnamese"], "GILL": ["sikh_punjabi"],
    # The four entries below have no forename-evidence sets, so they are
    # never assigned; they are listed to block deterministic assignment
    # (MOHAN additionally blocks the Indo-Caribbean suffix pattern).
    "RAI": ["nepali_himalayan", "indian"], "PAUL": ["indian"],
    "ROY": ["indian"], "MOHAN": ["indo_caribbean", "indian"],
}

KOREAN_GIVEN = set("""
SUNG SEUNG HYUN HYUNG JAE JOON KYUNG KYOUNG KWANG SOO SANG SEOK SUK HEE
MYUNG YOUNG YEON YONG EUN HYE JEONG WON TAE KYU BYUNG CHUL CHOL DUK DEOK
HWAN IL JONG JOO OK SEON WOOK YUL BONG BUM BEOM DAE GEUN KEUN HAK HYO
KYEONG SOON GYU MYEONG SEUL HWA
""".split())
CHINESE_GIVEN = set("""
WEI XIN JIAN MING QING LING PING ZHI GUO MEI XIAO QIANG JING YING ZHEN
ZHONG GANG HUA JIE KUN LEI NING PEI RONG SHAN SHENG SHU TAO TING WEN XIA
XIANG XIU XUE YAO YUAN YUE BIN BO CHAO FEI HAO HENG JIA KAI MIAO TIAN
QIU ZI FANG HUAN JUAN ZHAO XI ZHU CONG YIQING WAI KWOK KAM YUK SIU KIN
WING PUI SHUK SUET HING KWAI OI PIK SAU TAK CHEUK TSZ KA KIT LOK YAT HOI
QIN RUI SHUANG YU-
""".split()) - {"YU-"}
VIET_GIVEN = set("""
THI MINH HUONG PHUONG THANH TUAN ANH DUC HOA HUNG LINH NGOC NHU QUANG
THAO TRANG TRUNG TUYET XUAN HIEU KHANH LOAN TIEN VINH BICH CUONG DAT
DIEM DUNG DUY GIANG HANH LIEN LUAN NGA NGAN NGHIA NHAN OANH PHAT PHU
QUYEN QUOC THUY TOAN TRI UYEN VY HUE
""".split())
SIKH_GIVEN_RE = re.compile(r"(DEEP|JIT|PREET|INDER|WINDER|WANT)$")
GIVEN_SETS = {"korean": KOREAN_GIVEN, "chinese": CHINESE_GIVEN,
              "vietnamese": VIET_GIVEN}

ARTIFACTS = {"THE", "DE", "DEL", "LA", "VAN", "ESTATE", "TRUST", "BANK",
             "CHURCH", "CITY", "NEW", "SAINT", "ST"}


def forename_group(forename: str, candidates) -> str:
    toks = [t for t in re.split(r"[\s\-]+", forename or "") if t]
    hits = {}
    for g in candidates:
        if g == "sikh_punjabi":
            hits[g] = sum(bool(SIKH_GIVEN_RE.search(t)) for t in toks)
        elif g in GIVEN_SETS:
            hits[g] = sum(t in GIVEN_SETS[g] for t in toks)
        else:
            hits[g] = 0
    pos = {g: n for g, n in hits.items() if n > 0}
    if len(pos) == 1:
        return next(iter(pos))
    # a hit in a non-candidate set blocks assignment implicitly (only
    # candidates are scored; cross-set collisions were curated out)
    return ""


def build_classifier():
    import surgeo
    sm = surgeo.SurnameModel()
    tab = sm._PROB_RACE_GIVEN_SURNAME
    api = tab["api"]

    name_to_group = {}
    dropped = []
    for g, names in LISTS.items():
        for s in names:
            if s in ARTIFACTS or s in AMBIG:
                continue
            share = api.get(s, np.nan)
            if pd.notna(share) and share < API_FLOOR[g]:
                dropped.append((g, s, round(float(share), 3)))
                continue
            name_to_group[s] = g
    if dropped:
        print(f"dropped {len(dropped)} names below API floor:",
              sorted(dropped)[:20], "..." if len(dropped) > 20 else "")

    def classify(surname, forename):
        if not surname or surname in ARTIFACTS:
            return ""
        if surname in name_to_group:
            return name_to_group[surname]
        if surname in AMBIG:
            return forename_group(forename, AMBIG[surname])
        if IC_PATTERNS.search(surname) and surname not in IC_EXCLUDE:
            share = api.get(surname, np.nan)
            if TAMIL_RE.search(surname):
                if pd.notna(share) and share >= API_FLOOR["indian"]:
                    return "indian"
                return ""
            if pd.isna(share) or share >= API_FLOOR["indo_caribbean"]:
                return "indo_caribbean"
        return ""

    return classify, api


def load_panel():
    df = pd.read_csv(PANEL, dtype={"bct2020": str, "size_bin": str, "borocode": str})
    df["bbl_key"] = df["bbl_key"].astype(str)
    df = df[df["owner_type"] != "missing"]
    df["owner_occ_star"] = df["owner_occ_star"].astype(int)
    yb = df["yearbuilt"]
    df["era_pre1940"] = yb.between(1800, 1939).astype(int)
    df["era_4079"] = yb.between(1940, 1979).astype(int)
    df["era_8099"] = yb.between(1980, 1999).astype(int)
    df["era_unknown"] = (~yb.between(1800, 2026)).astype(int)
    df["multi_bldg"] = (df["numbldgs"] >= 2).astype(int)
    df["log2_area_per_unit"] = np.log2(df["area_per_unit"])
    for g in ["nyc_other", "outside_nyc", "unknown"]:
        df[f"geo_{g}"] = (df["owner_geo"] == g).astype(int)
    df["multi_prop_owner"] = df["multi_prop_owner"].astype(int)
    need = ["log2_area_per_unit", "value_rank", "size_bin", "bct2020"]
    df = df[df[need].notna().all(axis=1) & np.isfinite(df["log2_area_per_unit"])]
    bs = df[(df["owner_type"] == "individual") & (df["unitsres"] < 16)
            & df["p_white"].notna()].copy()
    print(f"BISG subsample: {len(bs):,}")
    return bs


def assign(bs):
    classify, api = build_classifier()
    nm = pd.read_csv(NAMES, dtype={"bbl_key": str})
    bs = bs.merge(nm, on="bbl_key", how="left")
    bs["surname"] = bs["surname"].fillna("")
    bs["forename"] = bs["forename"].fillna("")
    bs["subgroup"] = [classify(s, f) for s, f in zip(bs["surname"], bs["forename"])]
    bs.loc[(bs["surname"] == "SINGH"), "subgroup"] = "singh"

    bs["white_c"] = (bs["p_white"] > 0.7).astype(int)
    bs["black_c"] = (bs["p_black"] > 0.7).astype(int)
    bs["hisp_c"] = (bs["p_hispanic"] > 0.7).astype(int)
    bs["asian_uncl"] = ((bs["p_asian"] > 0.7) & (bs["subgroup"] == "")).astype(int)
    # subgroup dummies trump the coarse buckets
    for c in ["white_c", "black_c", "hisp_c", "asian_uncl"]:
        bs.loc[bs["subgroup"] != "", c] = 0

    counts = bs[bs["subgroup"] != ""].groupby("subgroup").size().sort_values(ascending=False)
    print("\nassigned properties by subgroup:")
    print(counts.to_string())
    print(f"unclassified high-p_asian: {bs['asian_uncl'].sum():,}; "
          f"classified white (reference): {bs['white_c'].sum():,}")

    top = (bs[bs["subgroup"] != ""].groupby(["subgroup", "surname"]).size()
           .rename("n").reset_index().sort_values(["subgroup", "n"], ascending=[True, False])
           .groupby("subgroup").head(8))
    print("\ntop surnames per subgroup:")
    for g, gg in top.groupby("subgroup"):
        print(f"  {g:<18}", ", ".join(f"{r.surname}({r.n})" for r in gg.itertuples()))

    mean_api = {g: round(float(np.mean([api.get(s, np.nan)
                for s in bs.loc[bs.subgroup == g, "surname"]])), 3)
                for g in counts.index}
    print("\nmean census API share of assigned names:", mean_api)

    bs[["bbl_key", "surname", "subgroup"]].to_csv(
        OUT / "asian_subgroup_assignments.csv", index=False)
    return bs


def acs_validation(bs):
    """Mean tract share of each detailed-Asian origin group (ACS B02015)
    and Guyanese/Trinidadian ancestry (B04006) at properties owned by each
    surname-classified subgroup."""
    import json
    import urllib.request

    env = Path.home() / "Dropbox/nycpol/ami-affordability-map/.env.local"
    key = ""
    for line in env.read_text().splitlines():
        if "CENSUS" in line.upper() and "=" in line:
            key = line.split("=", 1)[1].strip().strip('"')
    counties = ["005", "047", "061", "081", "085"]

    B02015 = {"B02015_021E": "acs_indian", "B02015_022E": "acs_bangladeshi",
              "B02015_002E": "acs_chinese", "B02015_012E": "acs_filipino",
              "B02015_004E": "acs_japanese", "B02015_005E": "acs_korean",
              "B02015_024E": "acs_nepalese", "B02015_025E": "acs_pakistani",
              "B02015_019E": "acs_vietnamese"}
    B04006 = {"B04006_045E": "acs_guyanese", "B04006_103E": "acs_trinidadian"}
    tot = "B01003_001E"

    frames = []
    for cty in counties:
        cols = ",".join([tot] + list(B02015) + list(B04006))
        url = (f"https://api.census.gov/data/2023/acs/acs5?get={cols}"
               f"&for=tract:*&in=state:36+county:{cty}&key={key}")
        with urllib.request.urlopen(url, timeout=120) as r:
            rows = json.load(r)
        d = pd.DataFrame(rows[1:], columns=rows[0])
        frames.append(d)
    acs = pd.concat(frames)
    for c in [tot] + list(B02015) + list(B04006):
        acs[c] = pd.to_numeric(acs[c], errors="coerce")
    acs = acs.rename(columns={**B02015, **B04006})
    boro = {"061": "1", "005": "2", "047": "3", "081": "4", "085": "5"}
    acs["bct2020"] = acs["county"].map(boro) + acs["tract"]
    share_cols = list(B02015.values()) + list(B04006.values())
    for c in share_cols:
        acs[c] = acs[c] / acs[tot].replace(0, np.nan) * 100

    m = bs.merge(acs[["bct2020"] + share_cols], on="bct2020", how="left")
    groups = [g for g in m.loc[m.subgroup != "", "subgroup"].unique()]
    rows = []
    for g in sorted(groups) + ["white_ref"]:
        sel = m["white_c"] == 1 if g == "white_ref" else m["subgroup"] == g
        rows.append({"subgroup": g, "n": int(sel.sum()),
                     **{c: round(float(m.loc[sel, c].mean()), 2) for c in share_cols}})
    val = pd.DataFrame(rows)
    val.to_csv(OUT / "asian_subgroup_validation.csv", index=False)
    print("\n== enclave validation: mean tract % of own/other origin groups ==")
    print(val.to_string(index=False))
    return val


def descriptives(bs):
    rows = []
    groups = sorted(bs.loc[bs.subgroup != "", "subgroup"].unique())
    for g in groups + ["white_ref", "asian_uncl"]:
        if g == "white_ref":
            sel = bs["white_c"] == 1
        elif g == "asian_uncl":
            sel = bs["asian_uncl"] == 1
        else:
            sel = bs["subgroup"] == g
        d = bs[sel]
        n = max(len(d), 1)
        rows.append({
            "subgroup": g, "n_props": len(d),
            "owner_occ": round(d["owner_occ_star"].mean() * 100, 1),
            "queens_share": round((d["borocode"] == "4").mean() * 100, 1),
            "brooklyn_share": round((d["borocode"] == "3").mean() * 100, 1),
            "compl_100": round(d["n_complaints"].sum() / n * 100, 1),
            "conv_100": round(d["n_conv"].sum() / n * 100, 1),
            "viol_100": round(d["n_viol_disp"].sum() / n * 100, 1),
            "ecb_100": round(d["n_ecb_2020on"].sum() / n * 100, 1),
        })
    desc = pd.DataFrame(rows)
    desc.to_csv(OUT / "asian_subgroup_descriptives.csv", index=False)
    print("\n== descriptives (raw, per 100 properties over 2020-May 2026) ==")
    print(desc.to_string(index=False))
    return desc


def models(bs):
    import pyfixest as pf
    vcov = {"CRV1": "bct2020"}
    groups = sorted(bs.loc[bs.subgroup != "", "subgroup"].unique())
    MIN_N = 700
    est_groups = [g for g in groups if (bs.subgroup == g).sum() >= MIN_N]
    small = [g for g in groups if g not in est_groups]
    if small:
        print(f"\nsubgroups below n={MIN_N}, kept as one 'asian_small' dummy: {small}")
    for g in est_groups:
        bs[f"sg_{g}"] = (bs["subgroup"] == g).astype(int)
    bs["sg_asian_small"] = bs["subgroup"].isin(small).astype(int)
    dummies = [f"sg_{g}" for g in est_groups] + (["sg_asian_small"] if small else []) + \
              ["asian_uncl", "black_c", "hisp_c"]
    bs["mixed_uncertain"] = ((bs[dummies].sum(axis=1) == 0) & (bs["white_c"] == 0)).astype(int)
    X = " + ".join(dummies + ["mixed_uncertain"] + BUILDING_COVARS + OWNER_COVARS)

    res = []

    def collect(m, model, outcome):
        t = m.tidy().reset_index()
        t.columns = [c.lower().replace(" ", "_").replace(".", "").replace("%", "pct")
                     for c in t.columns]
        t = t.rename(columns={t.columns[0]: "term"})
        t["model"], t["outcome"], t["n"] = model, outcome, m._N
        res.append(t)

    print("\n== PPML counts (vs classified-white reference) ==")
    for outc, label in [("n_complaints", "complaints"), ("n_conv", "conversion complaints"),
                        ("n_ecb_2020on", "ECB citations"), ("n_viol_disp", "disposition violations")]:
        m = pf.fepois(f"{outc} ~ {X} | size_bin + bct2020", data=bs, vcov=vcov)
        collect(m, f"ppml_{outc}", label)
        t = m.tidy()
        for g in est_groups:
            b, p = t.loc[f"sg_{g}", "Estimate"], t.loc[f"sg_{g}", "Pr(>|t|)"]
            print(f"  {label:<24} {g:<18} {(np.exp(b)-1)*100:+7.1f}%  (p {p:.4f})")

    # inspection-level margins
    conn = sqlite3.connect(str(config.DB_PATH))
    boro_case = """CASE b.borough WHEN 'MANHATTAN' THEN '1' WHEN 'BRONX' THEN '2'
        WHEN 'BROOKLYN' THEN '3' WHEN 'QUEENS' THEN '4' WHEN 'STATEN ISLAND' THEN '5' END"""
    c = pd.read_sql_query(f"""
        SELECT o.complaint_number, o.disposition_code, o.complaint_category,
               {boro_case} AS boro_code, b.block, b.lot
        FROM open_data o JOIN bis_scrape b USING(complaint_number)
        WHERE b.block IS NOT NULL AND b.lot IS NOT NULL
          AND substr(o.date_entered,7,4) >= '2020'""", conn)
    conn.close()
    c["bbl_key"] = [make_bbl(b, bl, lt) for b, bl, lt
                    in zip(c["boro_code"], c["block"], c["lot"])]
    c["outcome"] = c["disposition_code"].fillna("").astype(str).apply(classify_disposition)
    c["viol100"] = (c["outcome"] == "violation").astype(float) * 100
    c["noacc100"] = (c["outcome"] == "no_access").astype(float) * 100
    keep = ["bbl_key", "size_bin", "bct2020"] + dummies + ["mixed_uncertain", "white_c"] + \
        BUILDING_COVARS + OWNER_COVARS
    insp = c.merge(bs[keep], on="bbl_key")
    sub = insp[insp["outcome"].isin(["violation", "no_violation"])]
    print(f"\ninspection rows {len(insp):,}, substantive {len(sub):,}")
    print("== inspection-level LPMs (pp, within complaint code) ==")
    for frame, outc, label in [(sub, "viol100", "violation | substantive"),
                               (insp, "noacc100", "no access")]:
        m = pf.feols(f"{outc} ~ {X} | complaint_category + size_bin + bct2020",
                     data=frame, vcov=vcov)
        collect(m, f"lpm_{outc}", label)
        t = m.tidy()
        for g in est_groups:
            b, p = t.loc[f"sg_{g}", "Estimate"], t.loc[f"sg_{g}", "Pr(>|t|)"]
            print(f"  {label:<24} {g:<18} {b:+6.2f}pp (p {p:.4f})")

    pd.concat(res, ignore_index=True).to_csv(OUT / "asian_subgroup_estimates.csv", index=False)
    print(f"\nsaved -> {OUT/'asian_subgroup_estimates.csv'}")


def main():
    bs = load_panel()
    bs = assign(bs)
    acs_validation(bs)
    descriptives(bs)
    models(bs)


if __name__ == "__main__":
    main()

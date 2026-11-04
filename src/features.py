from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")
LAST_SEASON = 2013

LOCATIONS = ["San Francisco", "Berkeley Sunday", "Berkeley Saturday", "Peninsula",
             "Contra Costa", "Santa Rosa", "Orange County"]
SECTIONS = ["Orchestra", "Premium Orchestra", "Balcony Front", "Balcony", "Dress Circle"]
CITIES = ["san francisco", "berkeley", "oakland", "palo alto"]
ZERO_FILL_PREFIXES = ("sub_seats", "sub_tier_", "sub_n", "sub_total", "tix_n", "tix_seats", "tix_src",
                      "sub_streak", "sub_recency", "last_", "s13_", "sub_multi", "sub_frac_")


def season_start(season):
    return season.str[:4].astype(int)


def any_equal(values, target):
    return values.apply(lambda x: (x == target).any()).astype(int)


def streak(years, end):
    length = 0
    while end - length in years:
        length += 1
    return length


def load_data():
    def read(name, **kwargs):
        return pd.read_csv(DATA_DIR / name, **kwargs).rename(columns={"account.id": "id", "ID": "id"})

    return {
        "train": read("train.csv"),
        "test": read("test.csv"),
        "account": read("account.csv", encoding="latin1"),
        "subscriptions": read("subscriptions.csv"),
        "tickets": read("tickets_all.csv"),
        "zipcodes": pd.read_csv(DATA_DIR / "zipcodes.csv"),
    }


def subscription_features(subs):
    subs = subs.copy()
    subs["yr"] = season_start(subs["season"])
    subs["tier"] = subs["subscription_tier"].fillna(0)
    subs["seats"] = subs["no.seats"].fillna(0)
    subs["pl"] = subs["price.level"]

    g = subs.groupby("id")
    f = pd.DataFrame({
        "sub_n_rows": g.size(),
        "sub_n_seasons": g["yr"].nunique(),
        "sub_first_yr": g["yr"].min(),
        "sub_last_yr": g["yr"].max(),
        "sub_total_seats": g["seats"].sum(),
        "sub_mean_seats": g["seats"].mean(),
        "sub_total_tier": g["tier"].sum(),
        "sub_mean_pl": g["pl"].mean(),
        "sub_min_pl": g["pl"].min(),
        "sub_n_locations": g["location"].nunique(),
        "sub_n_packages": g["package"].nunique(),
        "sub_multi_any": any_equal(g["multiple.subs"], "yes"),
    })
    f["sub_tenure"] = f["sub_last_yr"] - f["sub_first_yr"] + 1
    f["sub_gap"] = LAST_SEASON - f["sub_last_yr"]
    f["sub_frac_seasons"] = f["sub_n_seasons"] / f["sub_tenure"]

    seats = subs.pivot_table(index="id", columns="yr", values="seats", aggfunc="sum").fillna(0)
    for y in range(2004, 2014):
        f[f"sub_seats_{y}"] = seats[y] if y in seats.columns else 0
    tiers = subs.pivot_table(index="id", columns="yr", values="tier", aggfunc="sum").fillna(0)
    for y in range(2009, 2014):
        f[f"sub_tier_{y}"] = tiers[y] if y in tiers.columns else 0

    years = g["yr"].apply(set)
    f["sub_streak_2013"] = years.apply(lambda s: streak(s, 2013))
    f["sub_streak_2012"] = years.apply(lambda s: streak(s, 2012))
    for window in (3, 5, 10):
        f[f"sub_n_last{window}"] = years.apply(lambda s, w=window: sum(LAST_SEASON - i in s for i in range(w)))
    f["sub_recency_w"] = subs.assign(w=0.7 ** (LAST_SEASON - subs["yr"])).groupby("id")["w"].sum()

    last = subs[subs["yr"] == g["yr"].transform("max")].groupby("id")
    f["last_pkg_full"] = any_equal(last["package"], "Full")
    f["last_pl"] = last["pl"].min()
    f["last_seats"] = last["seats"].sum()
    f["last_tier"] = last["tier"].sum()
    f["last_n_subs"] = last.size()
    for loc in LOCATIONS:
        f[f"last_loc_{loc[:6].replace(' ', '')}"] = any_equal(last["location"], loc)
    for sec in SECTIONS:
        f[f"last_sec_{sec.replace(' ', '')}"] = any_equal(last["section"], sec)

    s13 = subs[subs["yr"] == 2013].groupby("id")
    f["s13_pkg_full"] = any_equal(s13["package"], "Full")
    f["s13_pkg_quartet"] = s13["package"].apply(lambda x: x.str.contains("Quartet").any()).astype(int)
    f["s13_pkg_trio"] = s13["package"].apply(lambda x: x.str.contains("Trio").any()).astype(int)
    f["s13_pl"] = s13["pl"].min()
    f["s13_seats_vs_2012"] = f["sub_seats_2013"] - f["sub_seats_2012"]
    f["sub_tier_trend"] = f["sub_tier_2013"] - f["sub_tier_2011"]

    for package in ["Full", "Quartet", "Trio"]:
        f[f"sub_frac_{package}"] = subs.assign(m=subs["package"].str.startswith(package)).groupby("id")["m"].mean()
    return f


def ticket_features(tix):
    tix = tix.copy()
    tix["yr"] = season_start(tix["season"])
    tix["price.level"] = pd.to_numeric(tix["price.level"], errors="coerce")

    g = tix.groupby("id")
    f = pd.DataFrame({
        "tix_n": g.size(),
        "tix_seats": g["no.seats"].sum(),
        "tix_n_seasons": g["yr"].nunique(),
        "tix_last_yr": g["yr"].max(),
        "tix_mean_pl": g["price.level"].mean(),
        "tix_n_sets": g["set"].nunique(),
        "tix_multi": any_equal(g["multiple.tickets"], "yes"),
    })
    for y in range(2010, 2014):
        f[f"tix_n_{y}"] = tix[tix["yr"] == y].groupby("id").size()
    for source in tix["marketing.source"].dropna().value_counts().index[:6]:
        f[f"tix_src_{source[:10].replace(' ', '')}"] = tix[tix["marketing.source"] == source].groupby("id").size()
    return f


def account_features(acc, zipcodes):
    f = pd.DataFrame({
        "id": acc["id"],
        "don_2013": acc["amount.donated.2013"],
        "don_life": acc["amount.donated.lifetime"],
        "don_n": acc["no.donations.lifetime"],
        "don_first_yr": pd.to_datetime(acc["first.donated"], errors="coerce").dt.year,
    })
    f["don_life_log"] = np.log1p(f["don_life"])
    f["don_avg"] = f["don_life"] / f["don_n"].replace(0, np.nan)
    f["don_years"] = 2014 - f["don_first_yr"]

    relationship = acc["relationship"]
    f["has_rel"] = relationship.notna().astype(int)
    f["rel_edu"] = (relationship == "Education Participant").astype(int)
    f["rel_vendor"] = relationship.fillna("").str.contains("Vendor").astype(int)

    f["has_ship_zip"] = acc["shipping.zip.code"].notna().astype(int)
    f["has_bill_zip"] = acc["billing.zip.code"].notna().astype(int)
    f["same_zip"] = (acc["shipping.zip.code"] == acc["billing.zip.code"]).astype(int)
    zip5 = acc["billing.zip.code"].fillna(acc["shipping.zip.code"]).astype(str).str.extract(r"(\d{5})")[0]
    f["zip3"] = pd.to_numeric(zip5.str[:3], errors="coerce")

    city = acc["billing.city"].fillna(acc["shipping.city"]).fillna("").str.strip().str.lower()
    for c in CITIES:
        f[f"city_{c.replace(' ', '')}"] = (city == c).astype(int)

    z = zipcodes.dropna(subset=["Lat"]).drop_duplicates("Zipcode").copy()
    z["Zipcode"] = z["Zipcode"].astype(int)
    geo = pd.DataFrame({"zip": pd.to_numeric(zip5, errors="coerce")}).merge(
        z[["Zipcode", "Lat", "Long", "TotalWages", "TaxReturnsFiled"]],
        left_on="zip", right_on="Zipcode", how="left")
    f["lat"] = geo["Lat"].values
    f["lon"] = geo["Long"].values
    f["zip_wage_per_return"] = (geo["TotalWages"] / geo["TaxReturnsFiled"]).values
    f["dist_sf"] = np.hypot(f["lat"] - 37.78, (f["lon"] + 122.42) * 0.79)
    f["dist_berk"] = np.hypot(f["lat"] - 37.87, (f["lon"] + 122.26) * 0.79)
    return f


def build_features(data):
    ids = pd.concat([data["train"][["id"]], data["test"][["id"]]], ignore_index=True)
    f = (ids
         .merge(subscription_features(data["subscriptions"]), left_on="id", right_index=True, how="left")
         .merge(ticket_features(data["tickets"]), left_on="id", right_index=True, how="left")
         .merge(account_features(data["account"], data["zipcodes"]), on="id", how="left"))

    for col in f.columns:
        if col.startswith(ZERO_FILL_PREFIXES) and "yr" not in col:
            f[col] = f[col].fillna(0)

    f["is_sub_2013"] = (f["sub_seats_2013"] > 0).astype(int)
    f["ever_sub"] = (f["sub_n_rows"] > 0).astype(int)
    f["any_activity_recent"] = ((f["sub_n_last3"] > 0) | (f["tix_n"] > 0) | (f["don_2013"] > 0)).astype(int)

    train = data["train"].merge(f, on="id", how="left")
    test = data["test"].merge(f, on="id", how="left")
    return train, test


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(exist_ok=True)
    train, test = build_features(load_data())
    train.to_pickle(OUTPUT_DIR / "train_features.pkl")
    test.to_pickle(OUTPUT_DIR / "test_features.pkl")
    print(f"train: {train.shape}, test: {test.shape}")

"""
Generate comprehensive Indian car OBD + fault dataset.
Covers all major Indian market cars: Maruti, Hyundai, Tata, Honda,
Toyota, Mahindra, Kia, Renault, Nissan, Skoda, VW, MG, BMW, Audi,
Mercedes, Jeep, Ford (legacy), Fiat (legacy).

Output:
  data/indian_cars_obd.csv        - OBD readings + fault labels
  data/indian_cars_metadata.csv   - car specs + valuation data
  data/indian_cars_maintenance.csv - maintenance + failure labels
"""

import numpy as np
import pandas as pd
import random
import json
from datetime import datetime, timedelta

np.random.seed(42)
random.seed(42)

# ── All Indian market cars ─────────────────────────────────────────────────────

INDIAN_CARS = [
    # (brand, model, variant, fuel, segment, base_price_lakh, engine_cc)
    # MARUTI SUZUKI
    ("Maruti", "Alto K10",    "VXI",         "petrol", "hatchback", 3.99,  998),
    ("Maruti", "S-Presso",    "VXI+",        "petrol", "hatchback", 4.99,  998),
    ("Maruti", "Celerio",     "ZXI",         "petrol", "hatchback", 6.49,  998),
    ("Maruti", "WagonR",      "VXI 1.2",     "petrol", "hatchback", 6.99,  1197),
    ("Maruti", "WagonR",      "LXI CNG",     "cng",    "hatchback", 6.49,  1197),
    ("Maruti", "Swift",       "VXI",         "petrol", "hatchback", 6.49,  1197),
    ("Maruti", "Swift",       "ZXI+",        "petrol", "hatchback", 8.99,  1197),
    ("Maruti", "Dzire",       "VXI",         "petrol", "sedan",     7.49,  1197),
    ("Maruti", "Dzire",       "ZXI AMT",     "petrol", "sedan",     9.49,  1197),
    ("Maruti", "Baleno",      "Delta",       "petrol", "hatchback", 6.99,  1197),
    ("Maruti", "Baleno",      "Alpha",       "petrol", "hatchback", 9.49,  1197),
    ("Maruti", "Ignis",       "Zeta",        "petrol", "hatchback", 7.49,  1197),
    ("Maruti", "Ciaz",        "Alpha",       "petrol", "sedan",     9.49,  1462),
    ("Maruti", "Ertiga",      "VXI",         "petrol", "mpv",       8.69,  1462),
    ("Maruti", "Ertiga",      "ZXI CNG",     "cng",    "mpv",       10.49, 1462),
    ("Maruti", "XL6",         "Zeta+",       "petrol", "mpv",       11.49, 1462),
    ("Maruti", "Brezza",      "VXI",         "petrol", "suv",       8.49,  1462),
    ("Maruti", "Brezza",      "ZXI+",        "petrol", "suv",       13.99, 1462),
    ("Maruti", "Grand Vitara","Delta",       "petrol", "suv",       10.99, 1462),
    ("Maruti", "Grand Vitara","Zeta+ Hybrid","hybrid", "suv",       17.99, 1462),
    ("Maruti", "Fronx",       "Sigma",       "petrol", "suv",       7.51,  1197),
    ("Maruti", "Jimny",       "Zeta",        "petrol", "suv",       12.74, 1462),
    ("Maruti", "Invicto",     "Zeta+",       "hybrid", "mpv",       24.79, 1987),

    # HYUNDAI
    ("Hyundai", "Grand i10 Nios", "Magna",   "petrol", "hatchback", 6.47,  1197),
    ("Hyundai", "i20",         "Magna",      "petrol", "hatchback", 7.04,  1197),
    ("Hyundai", "i20",         "Asta",       "petrol", "hatchback", 10.99, 1197),
    ("Hyundai", "i20 N Line",  "N8 Dual",   "petrol", "hatchback", 12.49, 1197),
    ("Hyundai", "Aura",        "S CNG",      "cng",    "sedan",     8.49,  1197),
    ("Hyundai", "Verna",       "EX",         "petrol", "sedan",     10.99, 1497),
    ("Hyundai", "Verna",       "SX Turbo",   "petrol", "sedan",     15.99, 998),
    ("Hyundai", "Exter",       "S",          "petrol", "suv",       6.13,  1197),
    ("Hyundai", "Venue",       "S",          "petrol", "suv",       7.94,  1197),
    ("Hyundai", "Venue N Line","N8",         "petrol", "suv",       12.99, 998),
    ("Hyundai", "Creta",       "EX",         "petrol", "suv",       10.99, 1497),
    ("Hyundai", "Creta",       "SX Diesel",  "diesel", "suv",       18.99, 1493),
    ("Hyundai", "Creta N Line","N8",         "petrol", "suv",       19.99, 1497),
    ("Hyundai", "Alcazar",     "Platinum",   "petrol", "suv",       16.99, 1497),
    ("Hyundai", "Tucson",      "Platinum",   "diesel", "suv",       29.02, 1999),
    ("Hyundai", "Ioniq 5",     "Standard",   "ev",     "suv",       44.95, 0),
    ("Hyundai", "Kona Electric","S",         "ev",     "suv",       23.79, 0),

    # TATA
    ("Tata", "Tiago",          "XZ+",        "petrol", "hatchback", 5.60,  1199),
    ("Tata", "Tiago EV",       "Medium",     "ev",     "hatchback", 8.49,  0),
    ("Tata", "Tigor",          "XZ+",        "petrol", "sedan",     7.99,  1199),
    ("Tata", "Tigor EV",       "XZ+",        "ev",     "sedan",     12.49, 0),
    ("Tata", "Altroz",         "XZ+",        "petrol", "hatchback", 6.60,  1199),
    ("Tata", "Altroz",         "XZ Diesel",  "diesel", "hatchback", 9.49,  1497),
    ("Tata", "Altroz Racer",   "R3",         "petrol", "hatchback", 10.99, 1199),
    ("Tata", "Punch",          "Adventure",  "petrol", "suv",       6.13,  1199),
    ("Tata", "Punch EV",       "Medium",     "ev",     "suv",       10.99, 0),
    ("Tata", "Nexon",          "Smart+",     "petrol", "suv",       8.10,  1199),
    ("Tata", "Nexon",          "Creative+ S","diesel", "suv",       15.80, 1497),
    ("Tata", "Nexon EV",       "Medium",     "ev",     "suv",       14.49, 0),
    ("Tata", "Harrier",        "Smart",      "diesel", "suv",       14.99, 1956),
    ("Tata", "Harrier EV",     "Long Range", "ev",     "suv",       21.49, 0),
    ("Tata", "Safari",         "Smart",      "diesel", "suv",       15.49, 1956),
    ("Tata", "Curvv",          "Creative+",  "petrol", "suv",       10.00, 1199),
    ("Tata", "Curvv EV",       "Medium",     "ev",     "suv",       17.49, 0),

    # HONDA
    ("Honda", "Amaze",         "S CVT",      "petrol", "sedan",     7.99,  1199),
    ("Honda", "Amaze",         "VX Diesel",  "diesel", "sedan",     11.49, 1498),
    ("Honda", "City",          "V CVT",      "petrol", "sedan",     12.90, 1498),
    ("Honda", "City",          "ZX",         "petrol", "sedan",     15.99, 1498),
    ("Honda", "City e:HEV",    "ZX",         "hybrid", "sedan",     19.49, 1498),
    ("Honda", "Elevate",       "S",          "petrol", "suv",       11.00, 1498),
    ("Honda", "Elevate",       "ZX CVT",     "petrol", "suv",       16.99, 1498),
    ("Honda", "WR-V",          "VX",         "petrol", "suv",       10.99, 1199),

    # TOYOTA
    ("Toyota", "Glanza",       "G",          "petrol", "hatchback", 6.73,  1197),
    ("Toyota", "Rumion",       "G",          "petrol", "mpv",       10.49, 1462),
    ("Toyota", "Urban Cruiser Hyryder","G",  "petrol", "suv",       10.73, 1462),
    ("Toyota", "Urban Cruiser Hyryder","V Hybrid","hybrid","suv",   19.99, 1462),
    ("Toyota", "Innova Crysta","GX",         "diesel", "mpv",       19.99, 2393),
    ("Toyota", "Innova HyCross","G",         "petrol", "mpv",       18.99, 1987),
    ("Toyota", "Innova HyCross","ZX Hybrid", "hybrid", "mpv",       28.99, 1987),
    ("Toyota", "Fortuner",     "4x2 MT",     "diesel", "suv",       33.43, 2755),
    ("Toyota", "Fortuner Legender","4x4 AT", "diesel", "suv",       44.59, 2755),
    ("Toyota", "Hilux",        "Standard",   "diesel", "pickup",    30.40, 2393),
    ("Toyota", "Camry",        "Hybrid",     "hybrid", "sedan",     44.95, 2487),
    ("Toyota", "Vellfire",     "Premium",    "hybrid", "mpv",       139.99,2487),
    ("Toyota", "bZ4X",         "Premium",    "ev",     "suv",       54.99, 0),
    ("Toyota", "Land Cruiser", "VX",         "diesel", "suv",       210.0, 3346),

    # MAHINDRA
    ("Mahindra", "KUV100 NXT", "K8",         "petrol", "suv",       7.79,  1198),
    ("Mahindra", "Bolero",     "B4",         "diesel", "suv",       9.99,  1493),
    ("Mahindra", "Bolero Neo", "N8",         "diesel", "suv",       10.99, 1493),
    ("Mahindra", "XUV 3XO",    "MX1",        "petrol", "suv",       7.99,  1197),
    ("Mahindra", "XUV 3XO",    "AX7 L Turbo","petrol", "suv",       15.99, 1197),
    ("Mahindra", "XUV300",     "W4",         "petrol", "suv",       7.99,  1197),
    ("Mahindra", "XUV400 EV",  "EC Pro",     "ev",     "suv",       15.49, 0),
    ("Mahindra", "Scorpio-N",  "Z2",         "petrol", "suv",       13.99, 1997),
    ("Mahindra", "Scorpio-N",  "Z8 L Diesel","diesel", "suv",       23.99, 2184),
    ("Mahindra", "Scorpio Classic","S",      "diesel", "suv",       13.35, 2184),
    ("Mahindra", "XUV700",     "MX",         "petrol", "suv",       13.99, 1997),
    ("Mahindra", "XUV700",     "AX7 L Diesel","diesel","suv",       25.99, 2184),
    ("Mahindra", "Thar",       "LX 4x4 AT",  "petrol", "suv",       15.49, 1997),
    ("Mahindra", "Thar Roxx",  "MX1",        "petrol", "suv",       12.99, 1997),
    ("Mahindra", "BE 6e",      "Pack One",   "ev",     "suv",       18.90, 0),
    ("Mahindra", "XEV 9e",     "Pack One",   "ev",     "suv",       21.90, 0),
    ("Mahindra", "Marazzo",    "M6",         "diesel", "mpv",       13.88, 1497),
    ("Mahindra", "Supro",      "Profit Truck","diesel","commercial",6.34,  909),

    # KIA
    ("Kia", "Sonet",           "HTE",        "petrol", "suv",       7.99,  1197),
    ("Kia", "Sonet",           "GTX+ Diesel","diesel", "suv",       15.99, 1493),
    ("Kia", "Seltos",          "HTE",        "petrol", "suv",       10.89, 1497),
    ("Kia", "Seltos",          "GTX+ Diesel","diesel", "suv",       20.99, 1493),
    ("Kia", "Carens",          "Premium",    "petrol", "mpv",       10.49, 1497),
    ("Kia", "EV6",             "GT Line",    "ev",     "suv",       60.97, 0),
    ("Kia", "EV9",             "GT Line",    "ev",     "suv",       109.95,0),

    # RENAULT
    ("Renault", "Kwid",        "RXE",        "petrol", "hatchback", 4.70,  999),
    ("Renault", "Kwid",        "Climber",    "petrol", "hatchback", 5.90,  999),
    ("Renault", "Triber",      "RXE",        "petrol", "mpv",       6.00,  999),
    ("Renault", "Triber",      "RXZ AMT",    "petrol", "mpv",       8.99,  999),
    ("Renault", "Kiger",       "RXE",        "petrol", "suv",       6.00,  999),
    ("Renault", "Kiger",       "RXZ Turbo",  "petrol", "suv",       11.23, 999),

    # NISSAN
    ("Nissan", "Magnite",      "XE",         "petrol", "suv",       6.00,  999),
    ("Nissan", "Magnite",      "XV Turbo CVT","petrol","suv",       11.99, 999),

    # SKODA
    ("Skoda", "Slavia",        "Active",     "petrol", "sedan",     10.69, 1498),
    ("Skoda", "Slavia",        "Style DSG",  "petrol", "sedan",     17.99, 1498),
    ("Skoda", "Kushaq",        "Active",     "petrol", "suv",       10.89, 999),
    ("Skoda", "Kushaq",        "Style DSG",  "petrol", "suv",       18.99, 1498),
    ("Skoda", "Kodiaq",        "Sportline",  "petrol", "suv",       46.89, 1984),
    ("Skoda", "Superb",        "L&K",        "petrol", "sedan",     54.99, 1984),
    ("Skoda", "Octavia",       "Style",      "petrol", "sedan",     26.99, 1984),

    # VOLKSWAGEN
    ("Volkswagen", "Polo",     "Highline",   "petrol", "hatchback", 9.99,  999),
    ("Volkswagen", "Virtus",   "Comfortline","petrol", "sedan",     11.56, 1498),
    ("Volkswagen", "Virtus",   "GT DSG",     "petrol", "sedan",     17.99, 1498),
    ("Volkswagen", "Taigun",   "Comfortline","petrol", "suv",       11.56, 999),
    ("Volkswagen", "Taigun",   "GT DSG",     "petrol", "suv",       18.49, 1498),
    ("Volkswagen", "Tiguan",   "Elegance",   "petrol", "suv",       48.99, 1984),
    ("Volkswagen", "ID.4",     "Pro",        "ev",     "suv",       64.99, 0),

    # MG MOTOR
    ("MG", "Hector",           "Style",      "petrol", "suv",       13.99, 1451),
    ("MG", "Hector",           "Savvy Pro Diesel","diesel","suv",   19.99, 1956),
    ("MG", "Hector Plus",      "Select Pro", "petrol", "suv",       16.99, 1451),
    ("MG", "Astor",            "Style",      "petrol", "suv",       9.98,  1349),
    ("MG", "ZS EV",            "Excite",     "ev",     "suv",       18.98, 0),
    ("MG", "Comet EV",         "Excite",     "ev",     "hatchback", 7.98,  0),
    ("MG", "Gloster",          "Super",      "diesel", "suv",       34.99, 1996),
    ("MG", "Windsor EV",       "Excite",     "ev",     "suv",       13.50, 0),

    # JEEP
    ("Jeep", "Compass",        "Sport",      "diesel", "suv",       19.49, 1956),
    ("Jeep", "Compass",        "Limited 4x4","diesel", "suv",       27.99, 1956),
    ("Jeep", "Meridian",       "Limited",    "diesel", "suv",       29.99, 1956),
    ("Jeep", "Wrangler",       "Unlimited",  "petrol", "suv",       67.65, 1995),

    # BMW
    ("BMW", "2 Series Gran Coupe","220i M Sport","petrol","sedan",  37.90, 1998),
    ("BMW", "3 Series",        "330i M Sport","petrol", "sedan",    57.90, 1998),
    ("BMW", "5 Series",        "530i M Sport","petrol", "sedan",    68.90, 1998),
    ("BMW", "7 Series",        "740Li",      "petrol", "sedan",     172.0, 2998),
    ("BMW", "X1",              "sDrive18i",  "petrol", "suv",       46.90, 1499),
    ("BMW", "X3",              "xDrive20i",  "petrol", "suv",       69.90, 1998),
    ("BMW", "X5",              "xDrive40i",  "petrol", "suv",       93.90, 2998),
    ("BMW", "X7",              "xDrive40i",  "petrol", "suv",       122.50,2998),
    ("BMW", "iX",              "xDrive40",   "ev",     "suv",       111.0, 0),
    ("BMW", "i4",              "eDrive40",   "ev",     "sedan",     69.90, 0),
    ("BMW", "M3",              "Competition","petrol", "sedan",     142.0, 2993),

    # MERCEDES-BENZ
    ("Mercedes", "A-Class Limousine","A 200","petrol","sedan",      42.45, 1332),
    ("Mercedes", "C-Class",    "C 200",      "petrol", "sedan",     57.90, 1497),
    ("Mercedes", "E-Class",    "E 200",      "petrol", "sedan",     77.50, 1991),
    ("Mercedes", "S-Class",    "S 450 4MATIC","petrol","sedan",     222.0, 2999),
    ("Mercedes", "GLA",        "GLA 200",    "petrol", "suv",       50.50, 1332),
    ("Mercedes", "GLC",        "GLC 300",    "petrol", "suv",       73.50, 1999),
    ("Mercedes", "GLE",        "GLE 300d",   "diesel", "suv",       92.40, 1950),
    ("Mercedes", "GLS",        "GLS 450",    "petrol", "suv",       131.0, 2999),
    ("Mercedes", "EQS",        "EQS 580",    "ev",     "sedan",     162.0, 0),
    ("Mercedes", "EQB",        "EQB 350",    "ev",     "suv",       74.50, 0),
    ("Mercedes", "AMG C 63",   "S E Performance","hybrid","sedan",  165.0, 1991),

    # AUDI
    ("Audi", "A4",             "Premium Plus","petrol","sedan",     42.28, 1984),
    ("Audi", "A6",             "Premium Plus","petrol","sedan",     63.97, 1984),
    ("Audi", "A8 L",           "Technology", "petrol", "sedan",     159.97,2995),
    ("Audi", "Q3",             "Premium Plus","petrol","suv",       44.89, 1395),
    ("Audi", "Q5",             "Premium Plus","petrol","suv",       67.26, 1984),
    ("Audi", "Q7",             "Premium Plus","petrol","suv",       88.83, 2995),
    ("Audi", "Q8",             "Technology", "petrol", "suv",       130.65,2995),
    ("Audi", "e-tron",         "50",         "ev",     "suv",       99.99, 0),
    ("Audi", "e-tron GT",      "quattro",    "ev",     "sedan",     179.95,0),
    ("Audi", "RS5",            "Sportback",  "petrol", "sedan",     104.95,2894),

    # VOLVO
    ("Volvo", "XC40",          "Recharge",   "ev",     "suv",       55.90, 0),
    ("Volvo", "XC60",          "B5",         "petrol", "suv",       68.40, 1969),
    ("Volvo", "XC90",          "B5",         "petrol", "suv",       97.90, 1969),
    ("Volvo", "S90",           "B5",         "petrol", "sedan",     64.90, 1969),
    ("Volvo", "C40 Recharge",  "Twin",       "ev",     "suv",       61.25, 0),

    # LAND ROVER
    ("Land Rover", "Defender", "90",         "diesel", "suv",       79.56, 1997),
    ("Land Rover", "Discovery Sport","S",    "diesel", "suv",       67.46, 1997),
    ("Land Rover", "Range Rover Evoque","S", "diesel", "suv",       67.46, 1997),
    ("Land Rover", "Range Rover Sport","HSE","petrol", "suv",       119.96,2995),
    ("Land Rover", "Range Rover","SE",       "petrol", "suv",       231.0, 2995),

    # PORSCHE
    ("Porsche", "Cayenne",     "Base",       "petrol", "suv",       124.22,2894),
    ("Porsche", "Macan",       "Base",       "petrol", "suv",       83.93, 1984),
    ("Porsche", "Panamera",    "4S",         "petrol", "sedan",     196.84,2894),
    ("Porsche", "Taycan",      "Base",       "ev",     "sedan",     186.26,0),

    # ISUZU
    ("Isuzu", "D-Max",         "V-Cross",    "diesel", "pickup",    22.99, 1898),
    ("Isuzu", "MU-X",          "Base",       "diesel", "suv",       33.99, 1898),

    # CITROEN
    ("Citroen", "C3",          "Live",       "petrol", "hatchback", 6.16,  1199),
    ("Citroen", "C3 Aircross", "You",        "petrol", "suv",       9.99,  1199),
    ("Citroen", "eC3",         "Feel",       "ev",     "hatchback", 11.50, 0),

    # BYD
    ("BYD", "Atto 3",          "Standard",   "ev",     "suv",       33.99, 0),
    ("BYD", "Seal",            "Dynamic",    "ev",     "sedan",     41.99, 0),
    ("BYD", "e6",              "Base",       "ev",     "mpv",       29.15, 0),
]

print(f"Total car variants: {len(INDIAN_CARS)}")

# ── OBD parameter profiles per fuel type ──────────────────────────────────────

def obd_params_for_car(fuel, engine_cc, age_years, odometer_km, fault_level):
    """
    Generate realistic OBD readings based on fuel type, engine size,
    car age, odometer, and fault level (0=healthy, 1=minor, 2=major).
    """
    # Base values vary by fuel type
    base = {
        "petrol": {"rpm": 800, "coolant": 88,  "load": 28, "o2": 0.45, "maf": 4.5},
        "diesel": {"rpm": 750, "coolant": 85,  "load": 32, "o2": 0.0,  "maf": 5.2},
        "cng":    {"rpm": 820, "coolant": 86,  "load": 30, "o2": 0.42, "maf": 4.2},
        "hybrid": {"rpm": 650, "coolant": 82,  "load": 22, "o2": 0.48, "maf": 3.8},
        "ev":     {"rpm": 0,   "coolant": 35,  "load": 0,  "o2": 0.0,  "maf": 0.0},
    }.get(fuel, {"rpm": 800, "coolant": 88, "load": 28, "o2": 0.45, "maf": 4.5})

    # Age/odometer degradation
    deg = min(0.4, (age_years * 0.02) + (odometer_km / 500000))

    rpm     = base["rpm"] + np.random.normal(0, 50) + (fault_level * 120)
    coolant = base["coolant"] + np.random.normal(0, 3) + (fault_level * 8) + (deg * 15)
    load    = base["load"] + np.random.normal(0, 3) + (fault_level * 5)
    throttle= 15 + np.random.normal(0, 2)
    intake  = 32 + np.random.normal(0, 4)

    # Fuel trims — degrade with age and faults
    st_trim = np.random.normal(0, 2) + (fault_level * 4) + (deg * 3)
    lt_trim = np.random.normal(0, 1.5) + (fault_level * 6) + (deg * 5)

    # Battery voltage
    battery = 13.5 + np.random.normal(0, 0.2) - (fault_level * 0.8) - (deg * 0.5)
    if fuel == "ev": battery = 400 + np.random.normal(0, 5) - (fault_level * 20)

    o2     = max(0, base["o2"] + np.random.normal(0, 0.05))
    maf    = max(0, base["maf"] + np.random.normal(0, 0.3) + (fault_level * 0.8))
    timing = 15 + np.random.normal(0, 1.5) - (fault_level * 2)
    speed  = np.random.choice([0, 30, 60, 80, 100], p=[0.3, 0.2, 0.25, 0.15, 0.1])

    # DTC codes based on fault level
    dtc_list = []
    dtc_count = 0
    mil_on = False
    if fault_level == 1:
        possible = ["P0171", "P0420", "P0131", "P0401", "P0300"]
        if np.random.random() > 0.5:
            dtc_list = [random.choice(possible)]
            dtc_count = 1
            mil_on = True
    elif fault_level == 2:
        possible = ["P0300", "P0301", "P0302", "P0420", "P0172", "P0171"]
        n = np.random.randint(1, 4)
        dtc_list = random.sample(possible, min(n, len(possible)))
        dtc_count = len(dtc_list)
        mil_on = True

    return {
        "rpm":             max(0, round(rpm)),
        "speed":           int(speed),
        "coolant_temp":    round(min(130, max(-10, coolant)), 1),
        "intake_air_temp": round(intake, 1),
        "engine_load":     round(min(100, max(0, load)), 1),
        "throttle_pos":    round(min(100, max(0, throttle)), 1),
        "fuel_trim_st":    round(min(30, max(-30, st_trim)), 2),
        "fuel_trim_lt":    round(min(30, max(-30, lt_trim)), 2),
        "battery_voltage": round(min(16, max(8, battery)), 2),
        "o2_voltage":      round(min(1.2, max(0, o2)), 3),
        "maf":             round(max(0, maf), 2),
        "timing_advance":  round(timing, 1),
        "dtc_codes":       ",".join(dtc_list),
        "dtc_count":       dtc_count,
        "mil_on":          int(mil_on),
    }


# ── Generate OBD dataset ──────────────────────────────────────────────────────

print("Generating OBD + fault dataset...")
obd_rows = []

for brand, model, variant, fuel, segment, base_price, engine_cc in INDIAN_CARS:
    # Generate multiple records per car variant
    samples = 80 if fuel != "ev" else 40

    for _ in range(samples):
        year       = np.random.randint(2015, 2025)
        age_years  = 2025 - year
        # Odometer: Indian average 12,000-18,000 km/year
        odometer   = int(age_years * np.random.uniform(10000, 20000))
        city       = random.choice(["Mumbai", "Delhi", "Bangalore", "Hyderabad",
                                    "Pune", "Chennai", "Kolkata", "Ahmedabad",
                                    "Jaipur", "Surat", "Lucknow", "Chandigarh"])
        owner_type = random.choice(["first", "second", "third"])

        # Fault probability increases with age and odometer
        fault_prob = min(0.7, (age_years * 0.05) + (odometer / 300000))
        fault_rand = np.random.random()
        if fault_rand > fault_prob + 0.3:       fault_level = 0  # healthy
        elif fault_rand > fault_prob:           fault_level = 1  # minor
        else:                                   fault_level = 2  # major

        obd = obd_params_for_car(fuel, engine_cc, age_years, odometer, fault_level)

        row = {
            "brand": brand, "model": model, "variant": variant,
            "fuel_type": fuel, "segment": segment,
            "engine_cc": engine_cc, "year": year,
            "age_years": age_years, "odometer_km": odometer,
            "city": city, "owner_type": owner_type,
            "base_price_lakh": base_price,
            **obd,
            "fault_level":   fault_level,          # 0/1/2
            "is_faulty":     int(fault_level > 0),  # binary label
        }
        obd_rows.append(row)

obd_df = pd.DataFrame(obd_rows)
print(f"OBD dataset: {len(obd_df)} rows, {obd_df['is_faulty'].mean():.1%} faulty")


# ── Generate maintenance / failure dataset ────────────────────────────────────

print("Generating maintenance + failure prediction dataset...")
maint_rows = []

COMPONENT_FAILURE_KM = {
    "petrol": {"brake_pads": 40000, "tires": 50000, "battery": 60000,
               "transmission": 100000, "engine": 150000, "catalytic": 80000},
    "diesel": {"brake_pads": 45000, "tires": 55000, "battery": 55000,
               "transmission": 120000, "engine": 180000, "catalytic": 100000},
    "cng":    {"brake_pads": 38000, "tires": 48000, "battery": 50000,
               "transmission": 90000,  "engine": 130000, "catalytic": 70000},
    "ev":     {"brake_pads": 60000, "tires": 50000, "battery": 120000,
               "transmission": 200000, "engine": 200000, "catalytic": 200000},
    "hybrid": {"brake_pads": 55000, "tires": 52000, "battery": 100000,
               "transmission": 150000, "engine": 170000, "catalytic": 90000},
}

for brand, model, variant, fuel, segment, base_price, engine_cc in INDIAN_CARS:
    thresholds = COMPONENT_FAILURE_KM.get(fuel, COMPONENT_FAILURE_KM["petrol"])

    for _ in range(40):
        year      = np.random.randint(2012, 2025)
        age       = 2025 - year
        odometer  = int(age * np.random.uniform(8000, 22000))

        # Health score calculation
        base_health = max(20, 100 - (age * 2.5) - (odometer / 8000))
        health_score = int(np.clip(base_health + np.random.normal(0, 8), 10, 100))

        # Per-component health
        comp_health = {}
        for comp, km_threshold in thresholds.items():
            wear = min(1.0, odometer / km_threshold)
            comp_h = max(10, 100 - wear * 80 + np.random.normal(0, 5))
            comp_health[comp] = int(np.clip(comp_h, 10, 100))

        # Failure prediction labels (will fail within N km)
        needs_service_km = max(0, 10000 - (odometer % 10000))

        row = {
            "brand": brand, "model": model, "fuel_type": fuel,
            "segment": segment, "year": year, "age_years": age,
            "odometer_km": odometer, "base_price_lakh": base_price,
            "health_score": health_score,
            "engine_health": comp_health["engine"],
            "transmission_health": comp_health["transmission"],
            "battery_health": comp_health["battery"],
            "brake_health": comp_health["brake_pads"],
            "tire_health": comp_health["tires"],
            "needs_service_soon": int(needs_service_km < 2000),
            "needs_brake_replacement": int(comp_health["brake_pads"] < 35),
            "needs_battery_replacement": int(comp_health["battery"] < 40),
            "needs_tire_replacement": int(comp_health["tires"] < 30),
        }
        maint_rows.append(row)

maint_df = pd.DataFrame(maint_rows)
print(f"Maintenance dataset: {len(maint_df)} rows")


# ── Generate valuation dataset ────────────────────────────────────────────────

print("Generating Indian car valuation dataset...")
val_rows = []

CITY_MULTIPLIER = {
    "Mumbai": 1.12, "Delhi": 1.10, "Bangalore": 1.08,
    "Hyderabad": 1.05, "Pune": 1.04, "Chennai": 1.03,
    "Kolkata": 1.00, "Ahmedabad": 0.98, "Jaipur": 0.97,
    "Surat": 0.96, "Lucknow": 0.95, "Chandigarh": 0.99,
}

DEP_RATE = {"petrol": 0.15, "diesel": 0.18, "cng": 0.20,
            "hybrid": 0.12, "ev": 0.10}

for brand, model, variant, fuel, segment, base_price, engine_cc in INDIAN_CARS:
    dep_rate = DEP_RATE.get(fuel, 0.15)

    for _ in range(30):
        year      = np.random.randint(2010, 2025)
        age       = 2025 - year
        odometer  = int(age * np.random.uniform(8000, 22000))
        city      = random.choice(list(CITY_MULTIPLIER.keys()))
        owner_num = np.random.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        accidents = np.random.choice([0, 1, 2], p=[0.65, 0.25, 0.10])
        health    = max(20, 100 - age * 3 - odometer/10000 + np.random.normal(0, 5))
        damage_count = np.random.randint(0, 5)

        # Market value calculation
        dep = min(0.75, dep_rate * age)
        base_val = base_price * 100000  # convert lakh to rupees

        km_penalty = max(0, (odometer - 50000) / 10000) * 0.01
        owner_penalty = (owner_num - 1) * 0.05
        accident_penalty = accidents * 0.08
        health_adj = (health - 80) / 80 * 0.15
        city_mult = CITY_MULTIPLIER.get(city, 1.0)
        damage_penalty = damage_count * 0.03

        market_value = base_val * (1 - dep) * (1 - km_penalty) * \
                       (1 - owner_penalty) * (1 - accident_penalty) * \
                       (1 + health_adj) * city_mult * (1 - damage_penalty)
        market_value = max(50000, market_value + np.random.normal(0, 15000))

        row = {
            "brand": brand, "model": model, "fuel_type": fuel,
            "segment": segment, "year": year, "age_years": age,
            "engine_cc": engine_cc, "base_price_lakh": base_price,
            "odometer_km": odometer, "city": city,
            "owner_number": owner_num, "accidents": accidents,
            "health_score": int(np.clip(health, 10, 100)),
            "damage_count": damage_count,
            "market_value_inr": round(market_value, -2),
        }
        val_rows.append(row)

val_df = pd.DataFrame(val_rows)
print(f"Valuation dataset: {len(val_df)} rows")

# ── Save all datasets ─────────────────────────────────────────────────────────

obd_df.to_csv("data/indian_cars_obd.csv", index=False)
maint_df.to_csv("data/indian_cars_maintenance.csv", index=False)
val_df.to_csv("data/indian_cars_valuation.csv", index=False)

print("\n=== SAVED ===")
print(f"data/indian_cars_obd.csv          {len(obd_df):,} rows")
print(f"data/indian_cars_maintenance.csv  {len(maint_df):,} rows")
print(f"data/indian_cars_valuation.csv    {len(val_df):,} rows")
print(f"\nCars covered: {obd_df['brand'].nunique()} brands, "
      f"{obd_df[['brand','model']].drop_duplicates().shape[0]} models")
print(f"Fuel types: {sorted(obd_df['fuel_type'].unique())}")

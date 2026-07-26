import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, UnidentifiedImageError
from streamlit_cropper import st_cropper
from huggingface_hub import hf_hub_download

st.set_page_config(
    page_title="CaliPlant AI · Botanical Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="05_Assets/pngtree-california-poppy-flower-png-image_11487273.png"
)

HF_REPO_ID = "Sanchal-01/caliplant-model"
MODEL_FILENAME = "caliplant_v5_extented.keras"

PALETTES = {
    "dark": {
        "bg": "#070A12",
        "card": "#0A0F1C",
        "surface": "#131B2E",
        "border": "#232E45",
        "primary": "#10B981",
        "secondary": "#059669",
        "text_primary": "#F8FAFC",
        "text_secondary": "#CBD5E1",
        "text_muted": "#8C99AE",
        "danger": "#EF4444",
        "warning": "#F59E0B",
        "info": "#3B82F6",
        "success": "#22C55E",
        "btn_text": "#06281F",
        "alert_danger_text": "#FCA5A5",
        "alert_safe_text": "#86EFAC",
    },
    "light": {
        "bg": "#F7F5EF",
        "card": "#FFFFFF",
        "surface": "#F1EEE4",
        "border": "rgba(30,42,31,0.14)",
        "primary": "#3F7350",
        "secondary": "#2C5940",
        "text_primary": "#1E2A1F",
        "text_secondary": "#3A4A3D",
        "text_muted": "#5C6B5A",
        "danger": "#C0392B",
        "warning": "#A85A2A",
        "info": "#2563EB",
        "success": "#227A4A",
        "btn_text": "#F7F5EF",
        "alert_danger_text": "#9A2B20",
        "alert_safe_text": "#1F6B3D",
    },
}

if "theme_mode" not in st.session_state:
    st.session_state.theme_mode = "dark"

appearance = st.sidebar.radio(
    "APPEARANCE",
    ["Dark", "Light"],
    index=0 if st.session_state.theme_mode == "dark" else 1,
    horizontal=True,
)
st.session_state.theme_mode = appearance.lower()
_p = PALETTES[st.session_state.theme_mode]

CLASS_LABELS = [
    'Aloe_Vera', 'Bird_of_Paradise', 'Black_Sage', 'Bladderpod', 'Blue_Elderberry',
    'Blue_Gum_Eucalyptus', 'Bougainvillea', 'CA_Bush_Sunflower', 'California_Buckwheat',
    'California_Poppy', 'California_Sagebrush', 'Castor_Bean', 'Ceanothus_CA_Lilac',
    'Chalk_Dudleya', 'Coast_Live_Oak', 'Common_Fennel', 'English_Ivy', 'European_Olive',
    'Fiddle_Leaf_Fig', 'Flamingo_Flower', 'French_Lavender', 'Golden_Pothos',
    'Heartleaf_Philodendron', 'Ice_Plant', 'Indoor_Rose', 'Jacaranda', 'Jade_Plant',
    'Joshua_Tree', 'Manzanita', 'Mariposa_Lily', 'Matilija_Poppy', 'Meyer_Lemon',
    'Mustard_Plant', 'Our_Lords_Candle', 'Palo_Verde', 'Peace_Lily', 'Poison_Hemlock',
    'Prickly_Pear', 'Rosemary', 'Rubber_Plant', 'Snake_Plant', 'Spider_Plant',
    'Sticky_Monkey-flower', 'Stinknet', 'Swiss_Cheese_Plant', 'Toyon', 'Tree_Tobacco',
    'Western_Poison_Oak', 'Western_Sycamore', 'Wild_Radish', 'ZZ_Plant'
]

PLANT_METADATA = {
    'Aloe_Vera': {"common_name": "Aloe Vera", "scientific_name": "Aloe barbadensis miller", "native_status": "Non-Native", "water_needs": "Low", "fun_fact": "The clear gel inside the leaves contains over 75 active compounds and has been used for millennia to treat burns.", "toxicity_alert": "Safe for skin; toxic to pets if eaten."},
    'Bird_of_Paradise': {"common_name": "Bird of Paradise", "scientific_name": "Strelitzia reginae", "native_status": "Non-Native", "water_needs": "Moderate", "fun_fact": "So iconic to Southern California that it was officially named the City Flower of Los Angeles in 1952.", "toxicity_alert": "Mildly toxic to cats and dogs."},
    'Black_Sage': {"common_name": "Black Sage", "scientific_name": "Salvia mellifera", "native_status": "Native", "water_needs": "Low", "fun_fact": "This highly aromatic shrub is a favorite for honeybees; its honey is light-colored and peppery.", "toxicity_alert": "Non-toxic."},
    'Bladderpod': {"common_name": "Bladderpod", "scientific_name": "Peritoma arborea", "native_status": "Native", "water_needs": "Very Low", "fun_fact": "Its unique, inflated seed pods act like balloons; it blooms year-round in California.", "toxicity_alert": "Non-toxic, but has a strong, unpleasant smell."},
    'Blue_Elderberry': {"common_name": "Blue Elderberry", "scientific_name": "Sambucus cerulea", "native_status": "Native", "water_needs": "Moderate", "fun_fact": "Historically used to make flutes from hollow stems; berries are used in immune-boosting syrups.", "toxicity_alert": "Raw berries, leaves, and stems are toxic."},
    'Blue_Gum_Eucalyptus': {"common_name": "Blue Gum Eucalyptus", "scientific_name": "Eucalyptus globulus", "native_status": "Invasive", "water_needs": "Low", "fun_fact": "Sheds bark in long strips, creating oily litter that is highly combustible during fires.", "toxicity_alert": "Leaves and bark are toxic to pets and humans."},
    'Bougainvillea': {"common_name": "Bougainvillea", "scientific_name": "B. spectabilis", "native_status": "Non-Native", "water_needs": "Low", "fun_fact": "The vibrant petals are actually modified leaves called bracts; true flowers are tiny white trumpets inside.", "toxicity_alert": "Thorns are sharp; sap can cause skin irritation."},
    'CA_Bush_Sunflower': {"common_name": "CA Bush Sunflower", "scientific_name": "Encelia californica", "native_status": "Native", "water_needs": "Low", "fun_fact": "A fast-growing shrub with bright yellow flowers featuring a dark chocolate-brown center.", "toxicity_alert": "Non-toxic."},
    'California_Buckwheat': {"common_name": "California Buckwheat", "scientific_name": "Eriogonum fasciculatum", "native_status": "Native", "water_needs": "Very Low", "fun_fact": "Flowers transition from creamy white in spring to a striking rusty-red color in late summer.", "toxicity_alert": "Non-toxic."},
    'California_Poppy': {"common_name": "California Poppy", "scientific_name": "Eschscholzia californica", "native_status": "Native", "water_needs": "Very Low", "fun_fact": "Designated as the state flower in 1903; petals close at night or during cloudy weather.", "toxicity_alert": "Non-toxic to humans."},
    'California_Sagebrush': {"common_name": "California Sagebrush", "scientific_name": "Artemisia californica", "native_status": "Native", "water_needs": "Low", "fun_fact": "Nicknamed 'Cowboy Cologne' because travelers rubbed leaves on clothes to mask odors on trails.", "toxicity_alert": "Non-toxic."},
    'Castor_Bean': {"common_name": "Castor Bean", "scientific_name": "Ricinus communis", "native_status": "Invasive", "water_needs": "Low", "fun_fact": "Unprocessed seeds contain ricin, one of the most deadly natural poisons on Earth.", "toxicity_alert": "Highly Toxic: Seeds contain Ricin; fatal if chewed."},
    'Ceanothus_CA_Lilac': {"common_name": "Ceanothus (CA Lilac)", "scientific_name": "Ceanothus spp.", "native_status": "Native", "water_needs": "Very Low", "fun_fact": "Flowers contain saponins and can be rubbed with water to create a natural soap lather.", "toxicity_alert": "Non-toxic."},
    'Chalk_Dudleya': {"common_name": "Chalk Dudleya", "scientific_name": "Dudleya pulverulenta", "native_status": "Native", "water_needs": "Very Low", "fun_fact": "Leaves are covered in a white waxy powder acting as natural sunscreen against harsh coastal sun.", "toxicity_alert": "Non-toxic."},
    'Coast_Live_Oak': {"common_name": "Coast Live Oak", "scientific_name": "Quercus agrifolia", "native_status": "Native", "water_needs": "Low", "fun_fact": "Can live for centuries and provides essential habitat for over 100 species of birds and mammals.", "toxicity_alert": "Acorns are toxic to humans unless properly processed."},
    'Common_Fennel': {"common_name": "Common Fennel", "scientific_name": "Foeniculum vulgare", "native_status": "Invasive", "water_needs": "Low", "fun_fact": "Smells strongly of licorice when crushed; historically used for culinary and medicinal purposes.", "toxicity_alert": "Safe, but easily confused with toxic Hemlock."},
    'English_Ivy': {"common_name": "English Ivy", "scientific_name": "Hedera helix", "native_status": "Non-Native", "water_needs": "Moderate", "fun_fact": "NASA's Clean Air Study found it to be one of the best plants for filtering airborne mold particles.", "toxicity_alert": "Toxic to pets and can cause skin reactions in humans."},
    'European_Olive': {"common_name": "European Olive", "scientific_name": "Olea europaea", "native_status": "Non-Native", "water_needs": "Low", "fun_fact": "A symbol of peace; some trees live over 2,000 years and produce fruit in dry conditions.", "toxicity_alert": "Fruit safe after curing; pollen is a common allergen."},
    'Fiddle_Leaf_Fig': {"common_name": "Fiddle Leaf Fig", "scientific_name": "Ficus lyrata", "native_status": "Non-Native", "water_needs": "Moderate", "fun_fact": "Violin-shaped leaves can grow up to 12 inches wide, making it a favorite for interior design.", "toxicity_alert": "Sap can irritate skin; toxic to pets."},
    'Flamingo_Flower': {"common_name": "Flamingo Flower", "scientific_name": "Anthurium", "native_status": "Non-Native", "water_needs": "Moderate", "fun_fact": "Waxy red 'blooms' are among the longest-lasting on Earth, staying bright for months.", "toxicity_alert": "Toxic if ingested; causes mouth and throat irritation."},
    'French_Lavender': {"common_name": "French Lavender", "scientific_name": "Lavandula dentata", "native_status": "Non-Native", "water_needs": "Low", "fun_fact": "Essential oil scent has been scientifically shown to lower heart rate and reduce stress levels.", "toxicity_alert": "Toxic to cats and dogs if ingested."},
    'Golden_Pothos': {"common_name": "Golden Pothos", "scientific_name": "Epipremnum aureum", "native_status": "Non-Native", "water_needs": "Low to Med", "fun_fact": "Nicknamed 'Devil's Ivy' because it stays green in near-total darkness and is hard to kill.", "toxicity_alert": "Toxic to pets."},
    'Heartleaf_Philodendron': {"common_name": "Heartleaf Philodendron", "scientific_name": "Philodendron hederaceum", "native_status": "Non-Native", "water_needs": "Moderate", "fun_fact": "Translates from Greek as 'love tree' because of its classic heart-shaped leaves.", "toxicity_alert": "Toxic to cats and dogs if chewed."},
    'Ice_Plant': {"common_name": "Ice Plant", "scientific_name": "Carpobrotus edulis", "native_status": "Invasive", "water_needs": "Low", "fun_fact": "Introduced to stabilize railroad tracks, it now threatens coastal dunes by crowding native species.", "toxicity_alert": "Non-toxic, but environmentally harmful."},
    'Indoor_Rose': {"common_name": "Indoor Rose", "scientific_name": "Rosa chinensis", "native_status": "Non-Native", "water_needs": "Moderate", "fun_fact": "Miniature roses bred to stay small, representing a true rose experience in a compact size.", "toxicity_alert": "Non-toxic, but watch out for tiny thorns."},
    'Jacaranda': {"common_name": "Jacaranda", "scientific_name": "Jacaranda mimosifolia", "native_status": "Non-Native", "water_needs": "Moderate", "fun_fact": "Famous for turning entire city streets purple when dropping trumpet-shaped blossoms in spring.", "toxicity_alert": "Non-toxic."},
    'Jade_Plant': {"common_name": "Jade Plant", "scientific_name": "Crassula ovata", "native_status": "Non-Native", "water_needs": "Very Low", "fun_fact": "Called the 'Money Tree,' believed to bring good luck and financial prosperity to businesses.", "toxicity_alert": "Toxic to pets."},
    'Joshua_Tree': {"common_name": "Joshua Tree", "scientific_name": "Yucca brevifolia", "native_status": "Native", "water_needs": "Very Low", "fun_fact": "Named by Mormon settlers who thought branches resembled Joshua pointing toward the promised land.", "toxicity_alert": "Non-toxic."},
    'Manzanita': {"common_name": "Manzanita", "scientific_name": "Arctostaphylos", "native_status": "Native", "water_needs": "Very Low", "fun_fact": "Recognizable by smooth, mahogany-red bark; 'Manzanita' is Spanish for 'little apple'.", "toxicity_alert": "Non-toxic."},
    'Mariposa_Lily': {"common_name": "Mariposa Lily", "scientific_name": "Calochortus splendens", "native_status": "Native", "water_needs": "Low", "fun_fact": "Named after the Spanish word for 'butterfly,' intricate petal patterns mimic butterfly wings.", "toxicity_alert": "Non-toxic."},
    'Matilija_Poppy': {"common_name": "Matilija Poppy", "scientific_name": "Romneya coulteri", "native_status": "Native", "water_needs": "Very Low", "fun_fact": "Known as the 'Fried Egg Plant,' produces the largest flowers of any California native.", "toxicity_alert": "Non-toxic."},
    'Meyer_Lemon': {"common_name": "Meyer Lemon", "scientific_name": "Citrus meyeri", "native_status": "Non-Native", "water_needs": "Moderate", "fun_fact": "A hybrid between a lemon and mandarin orange, making it sweeter and thinner-skinned.", "toxicity_alert": "Safe for humans; skin oils are toxic to pets."},
    'Mustard_Plant': {"common_name": "Mustard Plant", "scientific_name": "Brassica nigra", "native_status": "Invasive", "water_needs": "Low", "fun_fact": "Releases chemicals into the soil that prevent native California plants from growing nearby.", "toxicity_alert": "Generally safe, but highly invasive."},
    'Our_Lords_Candle': {"common_name": "Our Lord's Candle", "scientific_name": "Hesperoyucca whipplei", "native_status": "Native", "water_needs": "Very Low", "fun_fact": "Sends up a massive 15-foot flower stalk; once blooming finishes, the entire plant dies.", "toxicity_alert": "Sharp leaf tips can cause physical injury."},
    'Palo_Verde': {"common_name": "Palo Verde", "scientific_name": "Parkinsonia florida", "native_status": "Native", "water_needs": "Very Low", "fun_fact": "Spanish for 'green stick'; green trunk and branches photosynthesize even when leaves drop.", "toxicity_alert": "Non-toxic."},
    'Peace_Lily': {"common_name": "Peace Lily", "scientific_name": "Spathiphyllum spp.", "native_status": "Non-Native", "water_needs": "Moderate", "fun_fact": "Blooms even in low light; known as the 'closet plant' for extreme dark resilience.", "toxicity_alert": "Toxic to pets; contains calcium oxalate crystals."},
    'Poison_Hemlock': {"common_name": "Poison Hemlock", "scientific_name": "Conium maculatum", "native_status": "Invasive", "water_needs": "Low to Med", "fun_fact": "Used to execute Socrates; identified by distinct purple splotches on smooth hollow stems.", "toxicity_alert": "Extremely Toxic: Fatal if any part is ingested."},
    'Prickly_Pear': {"common_name": "Prickly Pear", "scientific_name": "Opuntia ficus-indica", "native_status": "Native", "water_needs": "Very Low", "fun_fact": "Pads (Nopales) and sweet fruits (Tuna) are edible and popular in Mexican cuisine.", "toxicity_alert": "Watch out for glochids (microscopic irritating spines)."},
    'Rosemary': {"common_name": "Rosemary", "scientific_name": "Salvia rosmarinus", "native_status": "Non-Native", "water_needs": "Low", "fun_fact": "A hardy Mediterranean herb that thrives in dry weather; often used as a privacy hedge.", "toxicity_alert": "Edible and safe."},
    'Rubber_Plant': {"common_name": "Rubber Plant", "scientific_name": "Ficus elastica", "native_status": "Non-Native", "water_needs": "Low to Med", "fun_fact": "Grows up to 100-foot trees in the wild; indoor leaves look like polished rubber.", "toxicity_alert": "Toxic: Milky sap can irritate skin and eyes."},
    'Snake_Plant': {"common_name": "Snake Plant", "scientific_name": "Dracaena trifasciata", "native_status": "Non-Native", "water_needs": "Very Low", "fun_fact": "Converts CO2 into oxygen at night, making it ideal for bedroom air purification.", "toxicity_alert": "Toxic to pets if chewed."},
    'Spider_Plant': {"common_name": "Spider Plant", "scientific_name": "Chlorophytum comosum", "native_status": "Non-Native", "water_needs": "Moderate", "fun_fact": "Highly efficient at removing household toxins like formaldehyde and xylene from air.", "toxicity_alert": "Non-toxic and safe for pets."},
    'Sticky_Monkey-flower': {"common_name": "Sticky Monkey-flower", "scientific_name": "Diplacus aurantiacus", "native_status": "Native", "water_needs": "Low", "fun_fact": "Leaves have sticky resin preventing insect damage; roots were used to treat sores.", "toxicity_alert": "Non-toxic."},
    'Stinknet': {"common_name": "Stinknet", "scientific_name": "Oncosiphon piluliferum", "native_status": "Invasive", "water_needs": "Low", "fun_fact": "Smells like burnt rubber; dried flower globes are highly flammable fire hazards.", "toxicity_alert": "Can cause severe skin rashes and respiratory issues."},
    'Swiss_Cheese_Plant': {"common_name": "Swiss Cheese Plant", "scientific_name": "Monstera deliciosa", "native_status": "Non-Native", "water_needs": "Moderate", "fun_fact": "Holes in leaves allow sunlight and rain to pass through to lower jungle leaves.", "toxicity_alert": "Toxic to cats and dogs."},
    'Toyon': {"common_name": "Toyon", "scientific_name": "Heteromeles arbutifolia", "native_status": "Native", "water_needs": "Low", "fun_fact": "Called 'California Holly'; its abundance in hills led to the naming of Hollywood.", "toxicity_alert": "Berries contain cyanide precursors; toxic to pets."},
    'Tree_Tobacco': {"common_name": "Tree Tobacco", "scientific_name": "Nicotiana glauca", "native_status": "Invasive", "water_needs": "Low", "fun_fact": "Bright yellow tubular flowers attract hummingbirds, but the whole plant is lethal.", "toxicity_alert": "Toxic: Ingestion can be fatal to humans and animals."},
    'Western_Poison_Oak': {"common_name": "Western Poison Oak", "scientific_name": "Toxicodendron diversilobum", "native_status": "Native", "water_needs": "Low", "fun_fact": "Grows as a shrub, vine, or groundcover; oily urushiol stays active on gear for years.", "toxicity_alert": "High: Causes severe itchy rashes and blisters."},
    'Western_Sycamore': {"common_name": "Western Sycamore", "scientific_name": "Platanus racemosa", "native_status": "Native", "water_needs": "Moderate", "fun_fact": "Mottled bark peels off in patches, creating a natural camouflage look near creeks.", "toxicity_alert": "Non-toxic."},
    'Wild_Radish': {"common_name": "Wild Radish", "scientific_name": "Raphanus sativus", "native_status": "Invasive", "water_needs": "Low", "fun_fact": "Displays white, pink, and purple flowers—often all on the same individual plant.", "toxicity_alert": "Edible roots, but very pungent and bitter."},
    'ZZ_Plant': {"common_name": "ZZ Plant", "scientific_name": "Zamioculcas zamiifolia", "native_status": "Non-Native", "water_needs": "Very Low", "fun_fact": "Famous for being indestructible; survives in windowless low-light office corners.", "toxicity_alert": "Toxic if ingested."}
}

def inject_custom_css() -> None:
    css = """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,300;9..144,500;9..144,600;9..144,700&family=Inter:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        :root {
            --bg: __BG__;
            --card: __CARD__;
            --surface: __SURFACE__;
            --border: __BORDER__;
            --primary: __PRIMARY__;
            --secondary: __SECONDARY__;
            --text-primary: __TEXT_PRIMARY__;
            --text-secondary: __TEXT_SECONDARY__;
            --text-muted: __TEXT_MUTED__;
            --danger: __DANGER__;
            --warning: __WARNING__;
            --info: __INFO__;
            --success: __SUCCESS__;
            --btn-text: __BTN_TEXT__;
            --alert-danger-text: __ALERT_DANGER_TEXT__;
            --alert-safe-text: __ALERT_SAFE_TEXT__;
            --radius-sm: 10px;
            --radius-md: 14px;
            --radius-lg: 18px;
            --shadow-soft: 0 2px 14px rgba(0, 0, 0, 0.45);
        }

        html, body, [class*="css"] { font-family: 'Fraunces', serif; }

        .stApp { background: var(--bg); color: var(--text-primary); }
        #MainMenu, footer, header[data-testid="stHeader"] { background: transparent; }
        .block-container { padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1600px; }

        ::-webkit-scrollbar { width: 9px; height: 9px; }
        ::-webkit-scrollbar-track { background: var(--bg); }
        ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 8px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--primary); }

        @keyframes fadeInUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes softPulse { 0%, 100% { opacity: 0.55; } 50% { opacity: 1; } }
        @keyframes fillBar { from { width: 0%; } to { width: var(--fill-width); } }
        .fade-in { animation: fadeInUp 0.4s ease-out both; }

        .topbar {
            display: flex;
            justify-content: space-between;
            align-items: flex-end;
            padding-bottom: 1.1rem;
            margin-bottom: 1.6rem;
            border-bottom: 1px solid var(--border);
            flex-wrap: wrap;
            gap: 14px;
        }
        .topbar-title {
            font-family: 'Fraunces', serif;
            font-weight: 600;
            font-optical-sizing: auto;
            font-size: 44px;
            letter-spacing: 0.3px;
            color: var(--text-primary);
            margin: 0;
            line-height: 1;
        }
        .topbar-title .accent { color: var(--primary); font-style: normal; }
        .topbar-subtitle {
            font-family: 'Fraunces', serif;
            font-size: 13px;
            font-weight: 500;
            letter-spacing: 0.3px;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-top: 6px;
        }
        .status-pill {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            font-family: 'Fraunces', serif;
            font-size: 12.5px;
            font-weight: 500;
            letter-spacing: 0.3px;
            text-transform: uppercase;
            color: var(--primary);
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 7px 14px;
            border-radius: 100px;
            white-space: nowrap;
        }
        .status-pill .dot { width: 6px; height: 6px; border-radius: 50%; background: currentColor; box-shadow: 0 0 6px currentColor; animation: softPulse 1.6s ease-in-out infinite; }

        .step-label {
            font-family: 'Fraunces', serif;
            font-size: 12px;
            font-weight: 500;
            letter-spacing: 0.3px;
            text-transform: uppercase;
            color: var(--primary);
            margin-bottom: 4px;
        }
        .step-title {
            font-family: 'Fraunces', serif;
            font-size: 18.5px;
            font-weight: 600;
            color: var(--text-primary);
            margin-bottom: 1rem;
        }
        .section-title {
            font-family: 'Fraunces', serif;
            font-size: 18.5px;
            font-weight: 600;
            color: var(--text-primary);
            margin: 1.8rem 0 0.8rem 0;
            display: flex; align-items: center; gap: 0.5rem;
        }
        .section-title .section-line { flex: 1; height: 1px; background: var(--border); margin-left: 0.5rem; }

        .app-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 1.3rem 1.4rem;

        }

        .upload-caption { color: var(--text-muted); font-size: 0.86rem; margin: 0.6rem 0 0.7rem 0; }
        [data-testid="stFileUploaderDropzone"] {
            background: var(--card) !important;
            border: 1.5px dashed var(--border) !important;
            border-radius: var(--radius-md) !important;
            padding: 1rem !important;
            transition: border-color 0.25s ease, background 0.25s ease;
        }
        [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--primary) !important; background: rgba(16, 185, 129, 0.04) !important; }
        [data-testid="stFileUploaderDropzone"] button { background: var(--surface) !important; color: var(--text-primary) !important; border: 1px solid var(--border) !important; border-radius: var(--radius-sm) !important; }
        [data-testid="stFileUploaderDropzone"] button:hover { border-color: var(--primary) !important; color: var(--primary) !important; }
        [data-testid="stFileUploader"] section small { color: var(--text-muted) !important; }
        [data-testid="stWidgetLabel"] p { color: var(--text-secondary) !important; font-weight: 600; }
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] div,
        [data-testid="stFileUploaderDropzoneInstructions"] span,
        [data-testid="stFileUploaderDropzoneInstructions"] small {
            color: var(--text-secondary) !important;
        }

        div[role="radiogroup"] { gap: 6px; }
        div[role="radiogroup"] label {
            border: 1px solid var(--border);
            border-radius: 6px;
            padding: 4px 10px;
            background: var(--surface);
        }
        div[role="radiogroup"] label p { color: var(--text-primary) !important; }

        [data-testid="stSidebarCollapsedControl"] svg,
        [data-testid="stSidebarCollapsedControl"] button svg,
        [data-testid="stSidebarCollapsedControl"] button,
        [data-testid="stSidebarCollapseButton"] svg,
        [data-testid="stSidebarCollapseButton"] button,
        header[data-testid="stHeader"] svg {
            color: var(--text-primary) !important;
            fill: var(--text-primary) !important;
        }

        .stCheckbox label p { color: var(--text-muted) !important; font-size: 14px !important; font-family: 'Fraunces', serif; }

        [data-testid="stImage"] img {
            border-radius: calc(var(--radius-md) - 5px);
            border: 1px solid var(--border);
            background: var(--card);
            padding: 8px;
            box-sizing: border-box;
            margin-top: 0.9rem;
        }
        .image-caption { text-align: center; color: var(--text-muted); font-size: 0.8rem; margin-top: 0.5rem; }

        .empty-state {
            border: 1px dashed var(--border);
            border-radius: var(--radius-md);
            padding: 2.4rem 1.4rem;
            text-align: center;
            color: var(--text-muted);
            font-size: 0.9rem;
            height: 100%;
        }
        .empty-state .icon { font-size: 1.6rem; margin-bottom: 0.6rem; display: block; }

        .diagnosis-card {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 1.4rem 1.5rem;
            box-shadow: var(--shadow-soft);
        }
        .diagnosis-label {
            font-family: 'Fraunces', serif;
            font-size: 12px;
            font-weight: 500;
            letter-spacing: 0.3px;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
        }
        .diagnosis-name { font-family: 'Fraunces', serif; font-weight: 600; font-size: 27px; color: var(--text-primary); margin-bottom: 0.9rem; line-height: 1.2; }
        .confidence-badge { display: inline-flex; align-items: center; gap: 8px; font-family: 'Fraunces', serif; font-size: 12.5px; font-weight: 600; padding: 6px 15px; border-radius: 100px; letter-spacing: 0.3px; }
        .confidence-badge .dot { width: 7px; height: 7px; border-radius: 50%; background: currentColor; }
        .badge-success { background: rgba(34, 197, 94, 0.14); color: var(--success); border: 1px solid rgba(34, 197, 94, 0.35); }
        .badge-warning { background: rgba(245, 158, 11, 0.14); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.35); }
        .badge-danger  { background: rgba(239, 68, 68, 0.14); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.35); }

        .profile-panel {
            background: var(--card);
            border: 1px solid var(--border);
            border-radius: var(--radius-md);
            padding: 1.3rem 1.4rem;
            min-height: 150px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            text-align: center;
        }
        .profile-label { font-family: 'Fraunces', serif; font-size: 11.5px; font-weight: 500; letter-spacing: 0.3px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 0.7rem; text-align: center; }
        .profile-value-scientific { font-family: 'Fraunces', serif; font-style: italic; font-size: 0.98rem; color: var(--text-primary); font-weight: 600; line-height: 1.4; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; text-align: center; }
        .status-pill-sm { display: inline-flex; align-items: center; justify-content: center; width: fit-content; max-width: 100%; font-size: 0.85rem; font-weight: 700; padding: 5px 16px; border-radius: 100px; text-align: center; white-space: nowrap; }
        .pill-native    { background: rgba(34, 197, 94, 0.14); color: var(--success); border: 1px solid rgba(34, 197, 94, 0.35); }
        .pill-nonnative { background: rgba(245, 158, 11, 0.14); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.35); }
        .pill-invasive  { background: rgba(239, 68, 68, 0.14); color: var(--danger); border: 1px solid rgba(239, 68, 68, 0.35); }
        .pill-water     { background: rgba(59, 130, 246, 0.14); color: var(--info); border: 1px solid rgba(59, 130, 246, 0.35); }

        .callout-fact {
            border-radius: var(--radius-sm);
            padding: 0.95rem 1.3rem;
            font-size: 0.93rem;
            line-height: 1.55;
            border: 1px solid rgba(34, 197, 94, 0.32);
            background: linear-gradient(135deg, rgba(34, 197, 94, 0.10), rgba(16, 185, 129, 0.03));
            color: var(--alert-safe-text);
        }
        .callout-fact b { color: var(--text-primary); }

        .alert-box { border-radius: var(--radius-sm); padding: 0.95rem 1.3rem; font-size: 0.93rem; line-height: 1.55; border: 1px solid; }
        .alert-danger { background: rgba(239, 68, 68, 0.07); border-color: rgba(239, 68, 68, 0.32); color: var(--alert-danger-text); }
        .alert-safe   { background: rgba(34, 197, 94, 0.07); border-color: rgba(34, 197, 94, 0.32); color: var(--alert-safe-text); }
        .alert-box b { color: var(--text-primary); }

        .rank-card {
            display: flex; align-items: center; gap: 0.9rem;
            background: var(--card); border: 1px solid var(--border);
            border-radius: var(--radius-sm); padding: 0.75rem 1.1rem; margin-bottom: 0.6rem;
        }
        .rank-number {
            flex-shrink: 0; width: 30px; height: 30px; border-radius: 8px;
            display: flex; align-items: center; justify-content: center;
            font-family: 'Fraunces', serif; font-weight: 600; font-size: 0.85rem;
            background: var(--surface); color: var(--text-secondary); border: 1px solid var(--border);
        }
        .rank-1 { background: rgba(16, 185, 129, 0.14); color: var(--primary); border-color: rgba(16, 185, 129, 0.35); }
        .rank-body { flex: 1; min-width: 0; }
        .rank-top-row { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 0.4rem; gap: 0.5rem; }
        .rank-name { font-weight: 700; color: var(--text-primary); font-size: 0.92rem; }
        .rank-score { font-family: 'Fraunces', serif; font-weight: 600; color: var(--text-muted); font-size: 0.82rem; }
        .progress-track { width: 100%; height: 6px; background: var(--surface); border-radius: 100px; overflow: hidden; border: 1px solid var(--border); }
        .progress-fill { height: 100%; border-radius: 100px; background: linear-gradient(90deg, var(--primary), var(--secondary)); width: 0%; animation: fillBar 1.4s cubic-bezier(0.16, 1, 0.3, 1) forwards; }

        section[data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--border); }
        section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
        .sidebar-eyebrow { font-family: 'Fraunces', serif; font-size: 12px; font-weight: 500; letter-spacing: 0.3px; text-transform: uppercase; color: var(--text-muted); margin-bottom: 4px; }
        .sidebar-title { font-family: 'Fraunces', serif; font-weight: 600; font-size: 24px; color: var(--text-primary); margin-top: 2px; margin-bottom: 1.1rem; }
        .metric-row { display: flex; justify-content: space-between; align-items: center; padding: 0.55rem 0; border-bottom: 1px solid var(--border); }
        .metric-row:last-child { border-bottom: none; }
        .metric-label { font-size: 0.78rem; color: var(--text-muted); font-weight: 600; }
        .metric-value { font-size: 0.85rem; color: var(--text-primary); font-weight: 700; font-family: 'Fraunces', serif; }
        .sidebar-note { font-family: 'Fraunces', serif; font-size: 14px; font-weight: 500; letter-spacing: 0.3px; margin-top: 1.2rem; padding-top: 1rem; border-top: 1px solid var(--border); color: var(--text-muted); line-height: 1.7; }

        .stButton > button {
            width: 100%;
            background: linear-gradient(135deg, var(--primary), var(--secondary));
            color: var(--btn-text); font-weight: 700; font-size: 0.92rem;
            padding: 0.55rem 1.3rem; border-radius: var(--radius-sm); border: none;
            box-shadow: 0 3px 14px rgba(16, 185, 129, 0.22);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 6px 20px rgba(16, 185, 129, 0.35); color: var(--btn-text); }
        .stButton > button:active { transform: translateY(0); }
        .stButton > button:disabled { opacity: 0.6; box-shadow: none; }

        .stSpinner > div { text-align: center; color: var(--primary) !important; font-weight: 600; }

        .error-card { background: rgba(239, 68, 68, 0.06); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: var(--radius-md); padding: 1.1rem 1.3rem; color: var(--alert-danger-text); }
        .error-card b { color: var(--text-primary); }
        .error-card code { background: rgba(0,0,0,0.3); padding: 2px 8px; border-radius: 6px; color: var(--text-secondary); }

        .app-footer { text-align: center; color: var(--text-muted); font-size: 0.8rem; padding: 2rem 0 0.5rem 0; border-top: 1px solid var(--border); margin-top: 2.5rem; line-height: 1.6; }
        .app-footer .brand { color: var(--primary); font-weight: 700; }

        hr { border-color: var(--border) !important; }
        [data-testid="stImage"] img { border-radius: var(--radius-md); }

        @media (max-width: 900px) {
            .topbar { flex-direction: column; gap: 0.8rem; }
        }
        </style>
        """

    replacements = {
        "__BG__": _p["bg"],
        "__CARD__": _p["card"],
        "__SURFACE__": _p["surface"],
        "__BORDER__": _p["border"],
        "__PRIMARY__": _p["primary"],
        "__SECONDARY__": _p["secondary"],
        "__TEXT_PRIMARY__": _p["text_primary"],
        "__TEXT_SECONDARY__": _p["text_secondary"],
        "__TEXT_MUTED__": _p["text_muted"],
        "__DANGER__": _p["danger"],
        "__WARNING__": _p["warning"],
        "__INFO__": _p["info"],
        "__SUCCESS__": _p["success"],
        "__BTN_TEXT__": _p["btn_text"],
        "__ALERT_DANGER_TEXT__": _p["alert_danger_text"],
        "__ALERT_SAFE_TEXT__": _p["alert_safe_text"],
    }
    for token, value in replacements.items():
        css = css.replace(token, value)

    st.markdown(css, unsafe_allow_html=True)


@st.cache_resource
def load_caliplant_model(repo_id: str, filename: str):
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        token=st.secrets["HF_TOKEN"],
    )
    return tf.keras.models.load_model(local_path)


def preprocess_image(image: Image.Image) -> np.ndarray:
    image_rgb = image.convert("RGB")
    img_resized = image_rgb.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32)
    return np.expand_dims(img_array, axis=0)


def predict_image(model, img_batch: np.ndarray) -> np.ndarray:
    return model.predict(img_batch, verbose=0)[0]


def get_prediction_metadata(class_key: str) -> dict:
    return PLANT_METADATA.get(
        class_key,
        {
            "common_name": class_key.replace("_", " "),
            "scientific_name": "N/A",
            "native_status": "N/A",
            "water_needs": "N/A",
            "fun_fact": "No additional botanical details are available for this species.",
            "toxicity_alert": "N/A",
        },
    )


def get_confidence_style(confidence: float):
    if confidence >= 70.0:
        return "badge-success", "High Confidence"
    elif confidence >= 40.0:
        return "badge-warning", "Moderate Confidence"
    else:
        return "badge-danger", "Low Confidence"


def is_toxicity_warning(text: str) -> bool:
    keywords = ("toxic", "poison", "fatal", "danger", "severe")
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def render_top_bar() -> None:
    st.markdown(
        """
        <div class="topbar fade-in">
            <div>
                <h1 class="topbar-title">CaliPlant <span class="accent">AI</span></h1>
                <div class="topbar-subtitle">Botanical Intelligence Workspace</div>
            </div>
            <span class="status-pill"><span class="dot"></span> Model Ready</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-eyebrow"></div>
            <div class="sidebar-title">System Overview</div>
            """,
            unsafe_allow_html=True,
        )

        metrics = [
            ("Architecture", "MobileNetV2"),
            ("Species Coverage", "51 Classes"),
            ("Validation Accuracy", "86.09%"),
            ("Input Resolution", "224 × 224"),
        ]
        rows_html = "".join(
            f'<div class="metric-row"><span class="metric-label">{label}</span>'
            f'<span class="metric-value">{value}</span></div>'
            for label, value in metrics
        )
        st.markdown(f'<div class="app-card">{rows_html}</div>', unsafe_allow_html=True)

        st.markdown(
            """
            <div class="sidebar-note">
                Fine-tuned on a curated dataset of California native, ornamental,
                and invasive flora. Best results come from a clear, well-lit photo
                focused on a single leaf or flower.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_upload_header() -> None:
    st.markdown('<div class="step-label">Step 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Image Input</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="upload-caption">Drag &amp; drop a JPG, JPEG, or PNG photo of a leaf, flower, or full plant.</p>',
        unsafe_allow_html=True,
    )


def render_upload_body(uploaded_file) -> Image.Image | None:
    if uploaded_file is None:
        return None

    try:
        image = Image.open(uploaded_file)
        image.load()
    except (UnidentifiedImageError, OSError):
        render_error_card(
            "Invalid or Corrupted Image",
            "The uploaded file could not be read as a valid image. "
            "Please try a different <code>.jpg</code>, <code>.jpeg</code>, or <code>.png</code> file.",
        )
        return None

    use_crop = st.checkbox("Enable cropping workspace", value=False)
    st.markdown("<div style='margin-top: 0.6rem;'></div>", unsafe_allow_html=True)

    if use_crop:
        cropped_image = st_cropper(
            image,
            realtime_update=True,
            box_color="#127D45",
            aspect_ratio=None,
            return_type="image",
        )
        st.image(cropped_image, use_container_width=True)
        st.markdown('<div class="image-caption">Target crop segment</div>', unsafe_allow_html=True)
        return cropped_image
    else:
        st.image(image, use_container_width=True)
        st.markdown('<div class="image-caption">Uploaded source image</div>', unsafe_allow_html=True)
        return image


def render_empty_diagnostics() -> None:
    st.markdown('<div class="step-label">Step 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Diagnostics Report</div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="empty-state fade-in">
            Upload a photo and run the analysis to see the identification results here.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_diagnosis_card(common_name: str, confidence: float) -> None:
    st.markdown('<div class="step-label">Step 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="step-title">Diagnostics Report</div>', unsafe_allow_html=True)
    badge_class, label = get_confidence_style(confidence)
    st.markdown(
        f"""
        <div class="diagnosis-card fade-in">
            <div class="diagnosis-label">Top Prediction</div>
            <div class="diagnosis-name">{common_name}</div>
            <span class="confidence-badge {badge_class}">
                <span class="dot"></span> {confidence:.2f}% · {label}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if confidence < 40.0:
        st.markdown(
            """
            <div style="margin-top: 0.8rem;" class="callout-fact">
                <b>Tip:</b> This result has low confidence. Try a clearer, well-lit photo
                focused on the leaves or flowers for a more accurate identification.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_section_title(title: str) -> None:
    st.markdown(
        f'<div class="section-title">{title}<div class="section-line"></div></div>',
        unsafe_allow_html=True,
    )


def get_native_status_pill(status: str) -> str:
    s = status.lower()
    if "invasive" in s:
        css_class = "pill-invasive"
    elif "non-native" in s:
        css_class = "pill-nonnative"
    elif "native" in s:
        css_class = "pill-native"
    else:
        css_class = "pill-water"
    return f'<span class="status-pill-sm {css_class}">{status}</span>'


def render_botanical_profile(meta: dict) -> None:
    render_section_title("Botanical Profile")
    col1, col2, col3 = st.columns([1.5, 1, 1], gap="medium")
    with col1:
        st.markdown(
            f"""
            <div class="profile-panel fade-in">
                <div class="profile-label">Scientific Name</div>
                <div class="profile-value-scientific">{meta['scientific_name']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="profile-panel fade-in">
                <div class="profile-label">Native Status</div>
                {get_native_status_pill(meta['native_status'])}
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""
            <div class="profile-panel fade-in">
                <div class="profile-label">Water Needs</div>
                <span class="status-pill-sm pill-water">{meta['water_needs']}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_fun_fact(fun_fact: str) -> None:
    render_section_title("Fun Fact")
    st.markdown(
        f'<div class="callout-fact fade-in"><b>Did You Know:</b> {fun_fact}</div>',
        unsafe_allow_html=True,
    )


def render_toxicity_alert(toxicity_text: str) -> None:
    render_section_title("Toxicity Alert")
    if is_toxicity_warning(toxicity_text):
        st.markdown(
            f'<div class="alert-box alert-danger fade-in"><b>Caution Advised:</b> {toxicity_text}</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="alert-box alert-safe fade-in"><b>Safety Notice:</b> {toxicity_text}</div>',
            unsafe_allow_html=True,
        )


def render_top_predictions(predictions: np.ndarray) -> None:
    render_section_title("Top 3 Predictions")
    top3_indices = np.argsort(predictions)[-3:][::-1]

    for rank, idx in enumerate(top3_indices, start=1):
        class_key = CLASS_LABELS[idx]
        display_name = PLANT_METADATA.get(class_key, {}).get("common_name", class_key.replace("_", " "))
        score = float(predictions[idx] * 100)
        rank_class = "rank-1" if rank == 1 else ""
        fill_delay = (rank - 1) * 0.18
        card_delay = (rank - 1) * 0.1
        st.markdown(
            f"""
            <div class="rank-card fade-in" style="animation-delay: {card_delay}s;">
                <div class="rank-number {rank_class}">#{rank}</div>
                <div class="rank-body">
                    <div class="rank-top-row">
                        <span class="rank-name">{display_name}</span>
                        <span class="rank-score">{score:.2f}%</span>
                    </div>
                    <div class="progress-track">
                        <div class="progress-fill" style="--fill-width: {score}%; animation-delay: {fill_delay}s;"></div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_error_card(title: str, message: str) -> None:
    st.markdown(f'<div class="error-card fade-in"><b>{title}</b><br><br>{message}</div>', unsafe_allow_html=True)


def render_footer() -> None:
    st.markdown(
        """
        <div class="app-footer">
            CaliPlant AI is an AI model and can make mistakes — results may be inaccurate for blurry,
            poorly lit, or unfamiliar images. Verify with a botanical expert before acting on an identification.<br><br>
            <span class="brand">CaliPlant AI</span> · V5_Extended · Fine-Tuned Botanical Classifier
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    inject_custom_css()
    render_sidebar()
    render_top_bar()

    left_col, right_col = st.columns([2, 3], gap="large")

    with left_col:
        render_upload_header()
        uploaded_file = st.file_uploader(
            "Select an image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
        image = render_upload_body(uploaded_file)

        analyze_clicked = False
        if image is not None:
            st.markdown("<div style='margin-top: 1rem;'></div>", unsafe_allow_html=True)
            analyze_clicked = st.button("Run Diagnosis", use_container_width=True)

    upload_key = f"{uploaded_file.name}_{uploaded_file.size}" if uploaded_file is not None else None
    if upload_key != st.session_state.get("upload_key"):
        st.session_state["result"] = None
        st.session_state["upload_key"] = upload_key

    with right_col:
        if uploaded_file is None or image is None:
            render_empty_diagnostics()
            render_footer()
            return

        if analyze_clicked:
            with st.spinner("CaliPlant AI is analyzing botanical features..."):
                try:
                    model = load_caliplant_model(HF_REPO_ID, MODEL_FILENAME)
                except OSError:
                    render_error_card(
                        "Model File Missing",
                        f"The classification model could not be located at "
                        f"<code>{HF_REPO_ID}/{MODEL_FILENAME}</code>. Please verify the Hugging Face repo ID and filename.",
                    )
                    render_footer()
                    return
                except Exception as exc:
                    render_error_card(
                        "Model Loading Failed",
                        f"An unexpected error occurred while loading the model: <code>{exc}</code>",
                    )
                    render_footer()
                    return

                try:
                    img_batch = preprocess_image(image)
                    predictions = predict_image(model, img_batch)
                    top_idx = int(np.argmax(predictions))
                    top_confidence = float(predictions[top_idx] * 100)
                    top_class_key = CLASS_LABELS[top_idx]
                    meta = get_prediction_metadata(top_class_key)
                except Exception as exc:
                    render_error_card(
                        "Prediction Failed",
                        f"An unexpected error occurred while analyzing the image: <code>{exc}</code>",
                    )
                    render_footer()
                    return

                st.session_state["result"] = {
                    "predictions": predictions,
                    "meta": meta,
                    "confidence": top_confidence,
                }

        result = st.session_state.get("result")

        if result is None:
            render_empty_diagnostics()
            render_footer()
            return

        render_diagnosis_card(result["meta"]["common_name"], result["confidence"])
        render_botanical_profile(result["meta"])
        render_fun_fact(result["meta"]["fun_fact"])
        render_toxicity_alert(result["meta"]["toxicity_alert"])

    result = st.session_state.get("result")
    if result is not None:
        render_top_predictions(result["predictions"])

    render_footer()


if __name__ == "__main__":
    main()
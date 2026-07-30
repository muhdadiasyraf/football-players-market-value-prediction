import joblib
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="FIFA Player Market Value Predictor",
    layout="wide"
)

# Custom Styling for Valuation Display Card
st.markdown("""
    <style>
    .metric-card {
        background-color: #1E293B;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        border: 1px solid #334155;
        margin-top: 10px;
    }
    .metric-title {
        color: #94A3B8;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        color: #38BDF8;
        font-size: 2.8rem;
        font-weight: 700;
        margin-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)



# loading models and features
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('hist_gb_best_model.joblib')
        features = joblib.load('model_feature_columns.joblib')
        return model, features
    except Exception as e:
        st.error(f"Error loading model or feature artifacts: {e}")
        return None, None

model, FEATURE_COLUMNS = load_assets()

# header
st.title("FIFA Player Market Value Estimator")
st.write("Adjust the player information below to generate an estimated market valuation based on machine learning")
st.markdown("---")

if model is not None:

    # section 1: player information profile
    st.subheader("1. Profile & Primary Position")
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        age = st.slider("Age", min_value=16, max_value=40, value=24, step=1, help="Player's age in years.")
        international_reputation = st.selectbox("International Reputation (1-5 Star Rating)", [1, 2, 3, 4, 5], index=2)
    with col_p2:
        is_midfielder = st.checkbox("Is the player a Midfielder?", value=True, help="Check if the player operates primarily in midfield.")

    st.markdown("---")


    # section 2: attacking and technical skills
    st.subheader("2. Attacking & Playmaking")
    
    col_a1, col_a2, col_a3 = st.columns(3)
    
    with col_a1:
        ball_control = st.slider("Ball Control", 10, 99, 78)
        short_passing = st.slider("Short Passing", 10, 99, 76)
        dribbling = st.slider("Dribbling", 10, 99, 74)
        
    with col_a2:
        finishing = st.slider("Finishing", 10, 99, 70)
        positioning = st.slider("Positioning", 10, 99, 72)
        vision = st.slider("Vision", 10, 99, 73)
        
    with col_a3:
        crossing = st.slider("Crossing", 10, 99, 68)
        skill_moves = st.selectbox("Skill Moves (1-5 Star Rating)", [1, 2, 3, 4, 5], index=2)

    st.markdown("---")

    # section 3: defending and physical/mental traits
    st.subheader("3. Defending & Physicality / Mental")
    
    col_d1, col_d2, col_d3 = st.columns(3)
    
    with col_d1:
        standing_tackle = st.slider("Standing Tackle", 10, 99, 65)
        sliding_tackle = st.slider("Sliding Tackle", 10, 99, 60)
        interceptions = st.slider("Interceptions", 10, 99, 64)
        marking = st.slider("Marking", 10, 99, 62)
        
    with col_d2:
        sprint_speed = st.slider("Sprint Speed", 10, 99, 75)
        acceleration = st.slider("Acceleration", 10, 99, 74)
        stamina = st.slider("Stamina", 10, 99, 75)
        strength = st.slider("Strength", 10, 99, 70)
        
    with col_d3:
        reactions = st.slider("Reactions (Key Trait)", 10, 99, 80)
        composure = st.slider("Composure", 10, 99, 76)
        heading_accuracy = st.slider("Heading Accuracy", 10, 99, 65)

    st.markdown("---")

    # output
    if st.button("🚀 Calculate Estimated Market Value", type="primary", use_container_width=True):
        
        # dictionary matching with the exact features
        input_data = {
            'reactions': reactions,
            'age': age,
            'ball_control': ball_control,
            'short_passing': short_passing,
            'standing_tackle': standing_tackle,
            'composure': composure,
            'finishing': finishing,
            'heading_accuracy': heading_accuracy,
            'marking': marking,
            'dribbling': dribbling,
            'positioning': positioning,
            'sliding_tackle': sliding_tackle,
            'interceptions': interceptions,
            'sprint_speed': sprint_speed,
            'vision': vision,
            'strength': strength,
            'crossing': crossing,
            'stamina': stamina,
            'acceleration': acceleration,
            'skill_moves(1-5)': skill_moves,
            'international_reputation(1-5)': international_reputation,
            'pos_group_Midfielder': 1 if is_midfielder else 0,
        }


        # use the exact same feature sequence as the fitted model
        if hasattr(model, 'feature_names_in_'):
            expected_cols = list(model.feature_names_in_)
        else:
            expected_cols = FEATURE_COLUMNS

        # making dataframe match the exact feature and order
        input_df = pd.DataFrame(
            [[input_data.get(col, 0.0) for col in expected_cols]], 
            columns=expected_cols
        )

        # do prediction
        predicted_log = model.predict(input_df)[0]
        predicted_euro = np.expm1(predicted_log)

        # section 4: fifa player market value card
        st.markdown(f"""
            <div class="metric-card">
                <div class="metric-title">Estimated Market Value</div>
                <div class="metric-value">€{predicted_euro:,.2f}</div>
            </div>
        """, unsafe_allow_html=True)
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

from src.predict import predict_from_values


# USD to INR conversion rate
USD_TO_INR = 95.3


def predict_price(
    med_inc,
    house_age,
    ave_rooms,
    ave_bedrms,
    population,
    ave_occup,
    latitude,
    longitude,
):
    """Predict house price and display the result in Indian Rupees."""

    try:
        # Get prediction from the trained ML model.
        # predict_from_values() returns the price in USD.
        price_usd = predict_from_values(
            MedInc=med_inc,
            HouseAge=house_age,
            AveRooms=ave_rooms,
            AveBedrms=ave_bedrms,
            Population=population,
            AveOccup=ave_occup,
            Latitude=latitude,
            Longitude=longitude,
        )

        # Convert USD to INR
        price_inr = price_usd * USD_TO_INR

        return f"Predicted House Price: ₹{price_inr:,.2f}"

    except Exception as exc:
        return f"Prediction Error: {exc}"


# Create Gradio application
with gr.Blocks(title="House Price Prediction") as app:

    gr.Markdown(
        """
        # 🏠 House Price Prediction

        Enter the property characteristics below to predict
        the estimated house price using our trained
        Machine Learning model.

        **Currency:** Indian Rupees (₹)
        """
    )

    with gr.Row():

        with gr.Column():
            gr.Markdown("### 🏡 House Features")

            med_inc = gr.Number(
                label="Median Income",
                value=5.0,
            )

            house_age = gr.Number(
                label="House Age",
                value=20.0,
            )

            ave_rooms = gr.Number(
                label="Average Rooms",
                value=5.5,
            )

            ave_bedrms = gr.Number(
                label="Average Bedrooms",
                value=1.0,
            )

        with gr.Column():

            population = gr.Number(
                label="Population",
                value=1000.0,
            )

            ave_occup = gr.Number(
                label="Average Occupancy",
                value=3.0,
            )

            latitude = gr.Number(
                label="Latitude",
                value=34.05,
            )

            longitude = gr.Number(
                label="Longitude",
                value=-118.25,
            )

    predict_button = gr.Button(
        "🔮 Predict House Price",
        variant="primary",
    )

    prediction_output = gr.Textbox(
        label="Prediction",
        interactive=False,
    )

    predict_button.click(
        fn=predict_price,
        inputs=[
            med_inc,
            house_age,
            ave_rooms,
            ave_bedrms,
            population,
            ave_occup,
            latitude,
            longitude,
        ],
        outputs=prediction_output,
    )


if __name__ == "__main__":
    app.launch()
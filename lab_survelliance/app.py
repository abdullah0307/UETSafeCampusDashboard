# app.py

import streamlit as st

from lab_survelliance.utils import SurveillanceRepository, OverviewPage, PersonAnalyticsPage, RegionAnalyticsPage, \
    LiveCameraPage, SurveillanceService, FaceRegistrationPage
from utils.app_config import get_application_config
from utils.theme_reset import clear_persisted_theme_once


class LabSurveillanceApp:

    def __init__(self):
        config = get_application_config("lab_surveillance")

        repo = SurveillanceRepository(
            face_db_path=config["face_db_path"],
            activity_db_path=config["activity_db_path"],
        )

        service = SurveillanceService(repo)

        self.pages = {
            "📷 Live Camera": LiveCameraPage(service),
            "🧠 Face Registration": FaceRegistrationPage(service),
            "📊 Analytics and Report": OverviewPage(service),
            "👤 Person Analytics": PersonAnalyticsPage(service),
            "🗺️ Region Analytics": RegionAnalyticsPage(service),
        }

    def run(self):
        st.set_page_config(layout="wide")
        clear_persisted_theme_once()

        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Select Page", list(self.pages.keys()))

        self.pages[page].render()


if __name__ == "__main__":
    LabSurveillanceApp().run()

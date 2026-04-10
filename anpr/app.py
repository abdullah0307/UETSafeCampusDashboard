import streamlit as st

from anpr.utils import ConfigManager, DatabaseManager, PlateLogRepository, DashboardService, \
    RegisteredVehicleRepository, LiveMonitorPage, AnalyticsPage, VehicleSearchPage, CampusTimeTrackerPage, \
    RegisterVehiclePage
from utils.app_config import get_application_config
from utils.theme_reset import clear_persisted_theme_once


class ANPRDashboardApp:
    def __init__(self):
        config = ConfigManager(get_application_config("anpr"))
        db_manager = DatabaseManager(config.db_path)

        plate_repo = PlateLogRepository(db_manager)
        vehicle_repo = RegisteredVehicleRepository(db_manager)
        service = DashboardService(plate_repo, vehicle_repo)

        self.pages = {
            "📡 Live Camera View": LiveMonitorPage(plate_repo),
            "📊 Reports & Insights": AnalyticsPage(service),
            "🔍 Search Vehicles": VehicleSearchPage(service),
            "⏱️ Time Spent on Campus": CampusTimeTrackerPage(service),
            "📝 Manage Vehicle Registry": RegisterVehiclePage(vehicle_repo),
        }

    def run(self):
        st.set_page_config(layout="wide")
        clear_persisted_theme_once()
        st.sidebar.title("Navigation")
        page = st.sidebar.selectbox("Select Page", list(self.pages.keys()))
        self.pages[page].render()


if __name__ == "__main__":
    ANPRDashboardApp().run()

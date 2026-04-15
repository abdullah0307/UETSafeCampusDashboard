import streamlit as st
import base64

try:
    from st_click_detector import click_detector
except ImportError:
    click_detector = None

from anpr.app import ANPRDashboardApp
from attendance.app import AttendanceApp
from lab_survelliance.app import LabSurveillanceApp
from classroom_survelliance.app import ClassroomSurveillanceApp
from utils.app_config import get_application_config, load_app_config
from utils.db_initializer import initialize_all_databases
from utils.theme_reset import clear_persisted_theme_once


class SafeCampusLauncher:

    def __init__(self):
        self.app_configs = {
            "ANPR": get_application_config("anpr"),
            "Lab Surveillance": get_application_config("lab_surveillance"),
            "Classroom Surveillance": get_application_config("classroom_surveillance"),
            "Walk Through Attendance": get_application_config("attendance"),
        }
        self.projects = {
            "ANPR": ANPRDashboardApp(),
            "Lab Surveillance": LabSurveillanceApp(),
            "Classroom Surveillance": ClassroomSurveillanceApp(),
            "Walk Through Attendance": AttendanceApp(),
        }

    @staticmethod
    def _get_base64(image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _clickable_image(self, image_path, click_id, alt_text):
        if click_detector is None:
            st.warning(
                "Clickable image requires `st-click-detector`. "
                "Install with: pip install st-click-detector"
            )
            st.image(image_path, width='content')
            return False

        img_base64 = self._get_base64(image_path)
        content = f"""
            <div style='text-align:center; margin-top:6px;'>
                <a id='{click_id}' style='cursor:pointer;'>
                    <img src='data:image/png;base64,{img_base64}'
                         alt='{alt_text}'
                         style='max-width:100%; border-radius:20px; transition:all 0.3s ease;'
                         onmouseover='this.style.transform=\"translateY(-12px)\"; this.style.boxShadow=\"0 30px 60px rgba(0,0,0,0.7)\";'
                         onmouseout='this.style.transform=\"translateY(0)\"; this.style.boxShadow=\"none\";'>
                </a>
            </div>
        """
        return click_detector(content) == click_id

    def show_landing(self):
        st.markdown("""
        <style>

        html, body, [data-testid="stAppViewContainer"] {
            height: 100%;
            margin: 0;
            padding: 0;
        }

        .block-container {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }

        section[data-testid="stSidebar"] {
            display: none !important;
        }

        .fullscreen-center {
            position: fixed;
            inset: 0;
            display: flex;
            justify-content: center;
            align-items: center;
        }

        .center-content {
            text-align: center;
            width: 75%;
        }

        .main-title {
            font-size: 72px;
            font-weight: 700;
            margin-bottom: 70px;
        }

        .card-img {
            border-radius: 20px;
            transition: all 0.3s ease;
        }

        .card-img:hover {
            transform: translateY(-12px);
            box-shadow: 0 30px 60px rgba(0,0,0,0.7);
        }

        </style>
        """, unsafe_allow_html=True)

        ANPR_IMG = self.app_configs["ANPR"]["icon_path"]
        LAB_IMG = self.app_configs["Lab Surveillance"]["icon_path"]
        CLASS_IMG = self.app_configs["Classroom Surveillance"]["icon_path"]
        ATTENDANCE_IMG = self.app_configs["Walk Through Attendance"]["icon_path"]

        st.markdown('<div class="fullscreen-center">', unsafe_allow_html=True)
        st.markdown('<div class="center-content">', unsafe_allow_html=True)

        st.markdown('<div class="main-title">UET Safe Campus</div>', unsafe_allow_html=True)

        col1, col2, col3, col4 = st.columns(4)
        selected_project = None

        with col1:
            if self._clickable_image(ANPR_IMG, "anpr_card_click", "ANPR"):
                selected_project = "ANPR"

        with col2:
            if self._clickable_image(LAB_IMG, "lab_card_click", "Lab Surveillance"):
                selected_project = "Lab Surveillance"

        with col3:
            if self._clickable_image(CLASS_IMG, "class_card_click", "Classroom Surveillance"):
                selected_project = "Classroom Surveillance"

        with col4:
            if self._clickable_image(
                ATTENDANCE_IMG,
                "attendance_card_click",
                "Walk Through Attendance",
            ):
                selected_project = "Walk Through Attendance"

        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        if selected_project is not None:
            st.session_state.selected_project = selected_project
            st.rerun()

    def run(self):
        st.set_page_config(layout="wide")
        clear_persisted_theme_once()
        
        # Initialize all required databases
        config = load_app_config()
        db_results = initialize_all_databases(config)
        
        st.markdown(
            """
            <style>
                .stButton > button {
                    border-radius: 12px;
                    border: 1px solid rgba(255, 255, 255, 0.25);
                    background: linear-gradient(135deg, #1b2430 0%, #253447 100%);
                    color: #f4f7fb;
                    font-weight: 600;
                    transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
                    box-shadow: 0 10px 24px rgba(0, 0, 0, 0.25);
                }

                .stButton > button:hover {
                    transform: translateY(-2px);
                    border-color: rgba(120, 190, 255, 0.8);
                    box-shadow: 0 16px 28px rgba(0, 0, 0, 0.35), 0 0 0 1px rgba(120, 190, 255, 0.2) inset;
                }

                .stButton > button:active {
                    transform: translateY(0);
                    box-shadow: 0 8px 16px rgba(0, 0, 0, 0.35);
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        if "selected_project" not in st.session_state:
            st.session_state.selected_project = None

        if st.session_state.selected_project is None:
            st.markdown(
                """
                <style>
                    section[data-testid="stSidebar"] {display: none;}
                </style>
                """,
                unsafe_allow_html=True
            )
            self.show_landing()

        else:
            st.sidebar.markdown("---")
            if st.sidebar.button("⬅ Return To Launcher", width='stretch'):
                st.session_state.selected_project = None
                st.rerun()

            project = self.projects.get(st.session_state.selected_project)
            if project:
                project.run()


if __name__ == "__main__":
    SafeCampusLauncher().run()

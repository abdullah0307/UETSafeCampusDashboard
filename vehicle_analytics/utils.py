import os
import re
import sqlite3
from abc import ABC, abstractmethod
from datetime import datetime, time, timezone
from urllib.parse import urljoin, urlsplit

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
import yaml
from PIL import Image
import streamlit.components.v1 as components
from streamlit_autorefresh import st_autorefresh

from utils.app_config import get_application_config
from utils.pagination import PaginationManager


# ==============================
# CONFIG
# ==============================

class ConfigManager:
    def __init__(self, config=None):
        self._config = config or get_application_config("vehicle_analytics")

    @property
    def db_path(self):
        return self._config["database_path"]

    @property
    def live_frames_dir(self):
        return self._config["live_frames_dir"]

    @property
    def stream_source_mode(self):
        return self._config.get("stream_source_mode", "frame_files")

    @property
    def streams_api_url(self):
        return self._config.get("streams_api_url", "")

    @property
    def streams_public_base_url(self):
        return self._config.get("streams_public_base_url", "")


# ==============================
# DATABASE
# ==============================

class DatabaseManager:
    def __init__(self, db_path):
        self.db_path = db_path

    def connect(self):
        return sqlite3.connect(self.db_path, check_same_thread=False)


# ==============================
# REPOSITORIES
# ==============================

class PlateLogRepository:
    def __init__(self, db):
        self.db = db

    def get_cameras(self):
        with self.db.connect() as conn:
            df = pd.read_sql_query(
                "SELECT DISTINCT camera_id FROM plate_logs ORDER BY camera_id",
                conn,
            )
        return df["camera_id"].dropna().tolist()

    def search_logs(
            self,
            plate=None,
            camera=None,
            vehicle_type=None,
            plate_status=None,
            start=None,
            end=None,
            limit=None,
            page: int = 1,
            per_page: int = 50,
    ):
        query = """
            SELECT
                datetime(p.timestamp,'unixepoch','localtime') AS time,
                p.timestamp,
                p.camera_id,
                p.plate,
                p.confidence,
                p.vehicle_type,
                p.vehicle_image,
                p.plate_image,
                COALESCE(r.owner_name, 'Unknown') AS owner_name,
                p.plate_status,
                p.failure_reason,
                p.raw_ocr_text
            FROM plate_logs p
            LEFT JOIN registered_vehicles r
                ON UPPER(REPLACE(REPLACE(TRIM(p.plate),' ',''),'-','')) =
                   UPPER(REPLACE(REPLACE(TRIM(r.plate),' ',''),'-',''))
            WHERE 1=1
        """

        params = []

        # Plate filter
        if plate:
            query += " AND p.plate LIKE ?"
            params.append(f"%{plate}%")

        # Camera filter
        if camera:
            query += " AND p.camera_id = ?"
            params.append(camera)

        # Vehicle type filter
        if vehicle_type:
            query += " AND p.vehicle_type = ?"
            params.append(vehicle_type)

        # Plate status filter
        if plate_status:
            if plate_status == "Success":
                # READABLE indicates successful plate detection
                query += " AND UPPER(TRIM(p.plate_status)) = 'READABLE'"
            elif plate_status == "Failed":
                # Various failure states indicate failed plate detection
                query += " AND UPPER(TRIM(p.plate_status)) IN ('NO_LETTERS', 'NO_PLATE_DETECTED', 'TOO_SHORT', 'UNREADABLE', 'NO_DIGITS', 'CONTAINS_CHINESE', 'INVALID_FORMAT', 'TOO_LONG', 'EMPTY_OCR')"
            elif plate_status == "Partial":
                # Partial could be cases where plate was detected but with issues
                query += " AND UPPER(TRIM(p.plate_status)) NOT IN ('READABLE', 'NO_LETTERS', 'NO_PLATE_DETECTED', 'TOO_SHORT', 'UNREADABLE', 'NO_DIGITS', 'CONTAINS_CHINESE', 'INVALID_FORMAT', 'TOO_LONG', 'EMPTY_OCR') AND UPPER(TRIM(p.plate_status)) != ''"

        # Time filtering (STRICT UNIX comparison)
        if start is not None and end is not None:
            query += " AND p.timestamp BETWEEN ? AND ?"
            params.extend([start, end])

        elif start is not None:
            query += " AND p.timestamp >= ?"
            params.append(start)

        elif end is not None:
            query += " AND p.timestamp <= ?"
            params.append(end)

        # Order + limit/offset
        query += " ORDER BY p.timestamp DESC"

        # Support both legacy limit parameter and new pagination
        if limit is not None:
            query += " LIMIT ?"
            params.append(limit)
        else:
            offset = (page - 1) * per_page
            query += " LIMIT ? OFFSET ?"
            params.extend([per_page, offset])

        with self.db.connect() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            return df

    def search_logs_count(
            self,
            plate=None,
            camera=None,
            vehicle_type=None,
            plate_status=None,
            start=None,
            end=None,
    ):
        query = "SELECT COUNT(*) AS cnt FROM plate_logs p WHERE 1=1"
        params = []

        if plate:
            query += " AND p.plate LIKE ?"
            params.append(f"%{plate}%")

        if camera:
            query += " AND p.camera_id = ?"
            params.append(camera)

        if vehicle_type:
            query += " AND p.vehicle_type = ?"
            params.append(vehicle_type)

        # Plate status filter
        if plate_status:
            if plate_status == "Success":
                # READABLE indicates successful plate detection
                query += " AND UPPER(TRIM(p.plate_status)) = 'READABLE'"
            elif plate_status == "Failed":
                # Various failure states indicate failed plate detection
                query += " AND UPPER(TRIM(p.plate_status)) IN ('NO_LETTERS', 'NO_PLATE_DETECTED', 'TOO_SHORT', 'UNREADABLE', 'NO_DIGITS', 'CONTAINS_CHINESE', 'INVALID_FORMAT', 'TOO_LONG', 'EMPTY_OCR')"
            elif plate_status == "Partial":
                # Partial could be cases where plate was detected but with issues
                query += " AND UPPER(TRIM(p.plate_status)) NOT IN ('READABLE', 'NO_LETTERS', 'NO_PLATE_DETECTED', 'TOO_SHORT', 'UNREADABLE', 'NO_DIGITS', 'CONTAINS_CHINESE', 'INVALID_FORMAT', 'TOO_LONG', 'EMPTY_OCR') AND UPPER(TRIM(p.plate_status)) != ''"

        if start is not None and end is not None:
            query += " AND p.timestamp BETWEEN ? AND ?"
            params.extend([start, end])
        elif start is not None:
            query += " AND p.timestamp >= ?"
            params.append(start)
        elif end is not None:
            query += " AND p.timestamp <= ?"
            params.append(end)

        with self.db.connect() as conn:
            row = pd.read_sql_query(query, conn, params=params).iloc[0]
            return int(row["cnt"])

    def get_latest_log_timestamp(
            self,
            plate=None,
            camera=None,
            vehicle_type=None,
            plate_status=None,
            start=None,
            end=None,
    ):
        query = """
            SELECT MAX(p.timestamp) AS last_ts
            FROM plate_logs p
            WHERE 1=1
        """
        params = []

        if plate:
            query += " AND p.plate LIKE ?"
            params.append(f"%{plate}%")

        if camera:
            query += " AND p.camera_id = ?"
            params.append(camera)

        if vehicle_type:
            query += " AND p.vehicle_type = ?"
            params.append(vehicle_type)

        # Plate status filter
        if plate_status:
            if plate_status == "Success":
                # READABLE indicates successful plate detection
                query += " AND UPPER(TRIM(p.plate_status)) = 'READABLE'"
            elif plate_status == "Failed":
                # Various failure states indicate failed plate detection
                query += " AND UPPER(TRIM(p.plate_status)) IN ('NO_LETTERS', 'NO_PLATE_DETECTED', 'TOO_SHORT', 'UNREADABLE', 'NO_DIGITS', 'CONTAINS_CHINESE', 'INVALID_FORMAT', 'TOO_LONG', 'EMPTY_OCR')"
            elif plate_status == "Partial":
                # Partial could be cases where plate was detected but with issues
                query += " AND UPPER(TRIM(p.plate_status)) NOT IN ('READABLE', 'NO_LETTERS', 'NO_PLATE_DETECTED', 'TOO_SHORT', 'UNREADABLE', 'NO_DIGITS', 'CONTAINS_CHINESE', 'INVALID_FORMAT', 'TOO_LONG', 'EMPTY_OCR') AND UPPER(TRIM(p.plate_status)) != ''"

        if start is not None and end is not None:
            query += " AND p.timestamp BETWEEN ? AND ?"
            params.extend([start, end])
        elif start is not None:
            query += " AND p.timestamp >= ?"
            params.append(start)
        elif end is not None:
            query += " AND p.timestamp <= ?"
            params.append(end)

        with self.db.connect() as conn:
            row = conn.execute(query, params).fetchone()
            return row[0] if row and row[0] is not None else None

    def get_top_time_spent(self, entry_cam, exit_cam, start, end):
        query = """
            SELECT
                p.plate,
                MIN(CASE WHEN p.camera_id=? THEN p.timestamp END) entry_time,
                MAX(CASE WHEN p.camera_id=? THEN p.timestamp END) exit_time
            FROM plate_logs p
            WHERE datetime(p.timestamp,'unixepoch','localtime') BETWEEN ? AND ?
            GROUP BY p.plate
            HAVING entry_time IS NOT NULL AND exit_time IS NOT NULL
        """

        params = [entry_cam, exit_cam, start, end]

        with self.db.connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_vehicle_types(self):
        with self.db.connect() as conn:
            df = pd.read_sql_query(
                "SELECT DISTINCT vehicle_type FROM plate_logs ORDER BY vehicle_type",
                conn,
            )
        return df["vehicle_type"].dropna().tolist()

    def get_entry_exit(
            self,
            entry_cam,
            exit_cam,
            start_datetime,
            end_datetime,
            plate=None,
            vehicle_type=None,
            page: int = 1,
            per_page: int = 50,
    ):
        query = """
            SELECT plate, camera_id, timestamp, vehicle_image
            FROM plate_logs
            WHERE camera_id IN (?, ?)
              AND datetime(timestamp,'unixepoch','localtime') <= ?
        """

        params = [entry_cam, exit_cam, end_datetime]

        if plate:
            query += " AND plate LIKE ?"
            params.append(f"%{plate}%")

        if vehicle_type:
            query += " AND vehicle_type = ?"
            params.append(vehicle_type)

        query += " ORDER BY plate, timestamp"

        offset = (page - 1) * per_page
        query += " LIMIT ? OFFSET ?"
        params.extend([per_page, offset])

        with self.db.connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_entry_exit_count(
            self,
            entry_cam,
            exit_cam,
            start_datetime,
            end_datetime,
            plate=None,
            vehicle_type=None,
    ):
        query = """
            SELECT COUNT(*) AS cnt
            FROM plate_logs
            WHERE camera_id IN (?, ?)
              AND datetime(timestamp,'unixepoch','localtime') <= ?
        """

        params = [entry_cam, exit_cam, end_datetime]

        if plate:
            query += " AND plate LIKE ?"
            params.append(f"%{plate}%")

        if vehicle_type:
            query += " AND vehicle_type = ?"
            params.append(vehicle_type)

        with self.db.connect() as conn:
            row = pd.read_sql_query(query, conn, params=params).iloc[0]
            return int(row["cnt"])

    def get_latest_entry_exit_event_timestamp(
            self,
            entry_cam,
            exit_cam,
            end_datetime,
            plate=None,
            vehicle_type=None,
    ):
        query = """
            SELECT MAX(timestamp) AS last_ts
            FROM plate_logs
            WHERE camera_id IN (?, ?)
              AND datetime(timestamp,'unixepoch','localtime') <= ?
        """
        params = [entry_cam, exit_cam, end_datetime]

        if plate:
            query += " AND plate LIKE ?"
            params.append(f"%{plate}%")

        if vehicle_type:
            query += " AND vehicle_type = ?"
            params.append(vehicle_type)

        with self.db.connect() as conn:
            row = conn.execute(query, params).fetchone()
            return row[0] if row and row[0] is not None else None

    # def get_entry_exit(
    #         self,
    #         entry_cam,
    #         exit_cam,
    #         start_datetime,
    #         end_datetime,
    #         plate=None,
    # ):
    #     query = """
    #         SELECT
    #             p.plate,
    #             COALESCE(r.owner_name,'Unknown') owner_name,
    #             MIN(CASE WHEN p.camera_id=? THEN p.timestamp END) entry_time,
    #             MAX(CASE WHEN p.camera_id=? THEN p.timestamp END) exit_time,
    #             MAX(CASE WHEN p.camera_id=? THEN p.vehicle_image END) entry_image,
    #             MAX(CASE WHEN p.camera_id=? THEN p.vehicle_image END) exit_image
    #         FROM plate_logs p
    #         LEFT JOIN registered_vehicles r
    #             ON UPPER(REPLACE(REPLACE(TRIM(p.plate),' ',''),'-','')) =
    #                UPPER(REPLACE(REPLACE(TRIM(r.plate),' ',''),'-',''))
    #         WHERE datetime(p.timestamp,'unixepoch','localtime') BETWEEN ? AND ?
    #     """
    #
    #     params = [start_datetime, end_datetime]
    #
    #     if plate:
    #         query += " AND p.plate LIKE ?"
    #         params.append(f"%{plate}%")
    #
    #     query += """
    #         GROUP BY p.plate
    #         HAVING entry_time IS NOT NULL
    #            AND exit_time IS NOT NULL
    #         ORDER BY exit_time DESC
    #     """
    #
    #     params = [
    #                  entry_cam,
    #                  exit_cam,
    #                  entry_cam,
    #                  exit_cam,
    #                  start_datetime,
    #                  end_datetime,
    #              ] + (params[2:] if plate else [])
    #
    #     with self.db.connect() as conn:
    #         return pd.read_sql_query(query, conn, params=params)

    def get_total_count(self):
        with self.db.connect() as conn:
            return pd.read_sql_query(
                "SELECT COUNT(*) c FROM plate_logs",
                conn,
            ).iloc[0]["c"]

    def get_last_timestamp(self):
        with self.db.connect() as conn:
            return pd.read_sql_query(
                "SELECT MAX(timestamp) t FROM plate_logs",
                conn,
            ).iloc[0]["t"]

    def get_counts_by_camera(self):
        with self.db.connect() as conn:
            return pd.read_sql_query(
                """
                SELECT camera_id, COUNT(*) count
                FROM plate_logs
                GROUP BY camera_id
                """,
                conn,
            )

    def get_vehicle_type_summary(self):
        with self.db.connect() as conn:
            return pd.read_sql_query(
                """
                SELECT vehicle_type, COUNT(*) count
                FROM plate_logs
                GROUP BY vehicle_type
                """,
                conn,
            )

    def get_hourly_timeline(self):
        with self.db.connect() as conn:
            return pd.read_sql_query(
                """
                SELECT strftime('%Y-%m-%d %H:00', timestamp,'unixepoch','localtime') bucket,
                       COUNT(*) count
                FROM plate_logs
                GROUP BY bucket
                ORDER BY bucket
                """,
                conn,
            )

    def get_vehicle_type_timeline(self):
        with self.db.connect() as conn:
            return pd.read_sql_query(
                """
                SELECT strftime('%Y-%m-%d %H:00', timestamp,'unixepoch','localtime') bucket,
                       vehicle_type,
                       COUNT(*) count
                FROM plate_logs
                GROUP BY bucket, vehicle_type
                ORDER BY bucket
                """,
                conn,
            )

    def get_analytics_data(self, cameras=None, start=None, end=None):
        conditions = []
        params = []

        if cameras:
            placeholders = ",".join(["?"] * len(cameras))
            conditions.append(f"camera_id IN ({placeholders})")
            params.extend(cameras)

        if start and end:
            conditions.append(
                "date(datetime(timestamp,'unixepoch','localtime')) BETWEEN ? AND ?"
            )
            params.extend([start, end])

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        with self.db.connect() as conn:
            total = pd.read_sql_query(
                f"SELECT COUNT(*) c FROM plate_logs {where_clause}",
                conn,
                params=params,
            ).iloc[0]["c"]

            last_ts = pd.read_sql_query(
                f"SELECT MAX(timestamp) t FROM plate_logs {where_clause}",
                conn,
                params=params,
            ).iloc[0]["t"]

            cam_df = pd.read_sql_query(
                f"""
                SELECT camera_id, COUNT(*) count
                FROM plate_logs
                {where_clause}
                GROUP BY camera_id
                """,
                conn,
                params=params,
            )

            timeline_df = pd.read_sql_query(
                f"""
                SELECT strftime('%Y-%m-%d %H:00', timestamp,'unixepoch','localtime') bucket,
                       COUNT(*) count
                FROM plate_logs
                {where_clause}
                GROUP BY bucket
                ORDER BY bucket
                """,
                conn,
                params=params,
            )

            vehicle_summary_df = pd.read_sql_query(
                f"""
                SELECT vehicle_type, COUNT(*) count
                FROM plate_logs
                {where_clause}
                GROUP BY vehicle_type
                """,
                conn,
                params=params,
            )

            vehicle_timeline_df = pd.read_sql_query(
                f"""
                SELECT strftime('%Y-%m-%d %H:00', timestamp,'unixepoch','localtime') bucket,
                       vehicle_type,
                       COUNT(*) count
                FROM plate_logs
                {where_clause}
                GROUP BY bucket, vehicle_type
                ORDER BY bucket
                """,
                conn,
                params=params,
            )

        return total, last_ts, cam_df, timeline_df, vehicle_summary_df, vehicle_timeline_df

class RegisteredVehicleRepository:
    def __init__(self, db):
        self.db = db

    def ensure_table(self):
        with self.db.connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS registered_vehicles (
                    plate TEXT PRIMARY KEY,
                    owner_name TEXT NOT NULL,
                    notes TEXT,
                    created_at TEXT
                )
                """
            )
            conn.commit()

    def normalize(self, plate):
        return re.sub(r"[^A-Z0-9]", "", str(plate or "").upper())

    def exists(self, plate):
        norm = self.normalize(plate)
        with self.db.connect() as conn:
            row = conn.execute(
                """
                SELECT 1 FROM registered_vehicles
                WHERE UPPER(REPLACE(REPLACE(TRIM(plate),' ',''),'-','')) = ?
                """,
                (norm,),
            ).fetchone()
        return row is not None

    def register(self, plate, owner, notes):
        norm = self.normalize(plate)
        with self.db.connect() as conn:
            conn.execute(
                """
                INSERT INTO registered_vehicles
                (plate, owner_name, notes, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    norm,
                    owner,
                    notes,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
            conn.commit()

    def update(self, current_plate, new_plate, owner, notes):
        current_norm = self.normalize(current_plate)
        new_norm = self.normalize(new_plate) if new_plate else current_norm

        with self.db.connect() as conn:
            conn.execute(
                """
                UPDATE registered_vehicles
                SET plate = ?, owner_name = ?, notes = ?
                WHERE UPPER(REPLACE(REPLACE(TRIM(plate),' ',''),'-','')) = ?
                """,
                (new_norm, owner, notes, current_norm),
            )
            conn.commit()

    def delete(self, plate):
        norm = self.normalize(plate)
        with self.db.connect() as conn:
            deleted = conn.execute(
                """
                DELETE FROM registered_vehicles
                WHERE UPPER(REPLACE(REPLACE(TRIM(plate),' ',''),'-','')) = ?
                """,
                (norm,),
            ).rowcount
            conn.commit()
        return deleted > 0

    def search(self, plate=None, owner=None, page: int = 1, per_page: int = 50):
        query = "SELECT * FROM registered_vehicles WHERE 1=1"
        params = []

        if plate:
            query += " AND plate LIKE ?"
            params.append(f"%{plate.upper()}%")

        if owner:
            query += " AND owner_name LIKE ?"
            params.append(f"%{owner}%")

        query += " ORDER BY created_at DESC"

        offset = (page - 1) * per_page
        query += " LIMIT ? OFFSET ?"
        params.extend([per_page, offset])

        with self.db.connect() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def search_count(self, plate=None, owner=None):
        query = "SELECT COUNT(*) AS cnt FROM registered_vehicles WHERE 1=1"
        params = []

        if plate:
            query += " AND plate LIKE ?"
            params.append(f"%{plate.upper()}%")

        if owner:
            query += " AND owner_name LIKE ?"
            params.append(f"%{owner}%")

        with self.db.connect() as conn:
            row = pd.read_sql_query(query, conn, params=params).iloc[0]
            return int(row["cnt"])


# ==============================
# SERVICES
# ==============================

class DashboardService:
    def __init__(self, plate_repo, vehicle_repo):
        self.plate_repo = plate_repo
        self.vehicle_repo = vehicle_repo

    def get_top_time_spent_data(self, entry_cam, exit_cam, start, end):
        df = self.plate_repo.get_top_time_spent(entry_cam, exit_cam, start, end)

        if df.empty:
            return df

        df["entry_time"] = pd.to_datetime(df["entry_time"], unit="s")
        df["exit_time"] = pd.to_datetime(df["exit_time"], unit="s")

        df["duration_minutes"] = (
                (df["exit_time"] - df["entry_time"])
                .dt.total_seconds()
                .abs() / 60
        )

        df = df.sort_values("duration_minutes", ascending=False)

        return df


    def get_campus_time_data(
            self,
            entry_cam,
            exit_cam,
            start,
            end,
            plate=None,
            vehicle_type=None,
    ):
        df = self.plate_repo.get_entry_exit(
            entry_cam,
            exit_cam,
            start,
            end,
            plate,
            vehicle_type,
        )

        if df.empty:
            return df

        sessions = []

        for plate_value, group in df.groupby("plate"):

            group = group.sort_values("timestamp")
            entry_time = None
            entry_image = None

            for _, row in group.iterrows():

                if row["camera_id"] == entry_cam:
                    entry_time = row["timestamp"]
                    entry_image = row["vehicle_image"]

                elif row["camera_id"] == exit_cam and entry_time is not None:

                    if row["timestamp"] > entry_time:
                        duration_sec = row["timestamp"] - entry_time

                        sessions.append({
                            "plate": plate_value,
                            "entry_timestamp": entry_time,
                            "exit_timestamp": row["timestamp"],
                            "duration_seconds": duration_sec,
                            "duration_minutes": round(duration_sec / 60, 2),
                            "duration_hms": format_duration(duration_sec),
                            "entry_image": entry_image,
                            "exit_image": row["vehicle_image"],
                        })

                    entry_time = None
                    entry_image = None

        result_df = pd.DataFrame(sessions)

        if result_df.empty:
            return result_df

        local_tz = datetime.now().astimezone().tzinfo
        result_df["entry_time"] = (
            pd.to_datetime(result_df["entry_timestamp"], unit="s", utc=True)
            .dt.tz_convert(local_tz)
            .dt.tz_localize(None)
        )
        result_df["exit_time"] = (
            pd.to_datetime(result_df["exit_timestamp"], unit="s", utc=True)
            .dt.tz_convert(local_tz)
            .dt.tz_localize(None)
        )

        start_dt = datetime.strptime(start, "%Y-%m-%d %H:%M:%S")
        end_dt = datetime.strptime(end, "%Y-%m-%d %H:%M:%S")
        max_session_seconds = 24 * 60 * 60

        # Keep completed visits fully inside the selected window.
        # Cap very long sessions that usually come from unmatched/missing events.
        result_df = result_df[
            (result_df["entry_time"] >= start_dt)
            & (result_df["entry_time"] <= end_dt)
            & (result_df["exit_time"] >= start_dt)
            & (result_df["exit_time"] <= end_dt)
            & (result_df["duration_seconds"] <= max_session_seconds)
        ].copy().drop(columns=["entry_timestamp", "exit_timestamp"])

        if result_df.empty:
            return result_df

        # Attach owner_name
        owners = self.vehicle_repo.search()
        owners_dict = dict(zip(owners["plate"], owners["owner_name"]))

        result_df["owner_name"] = result_df["plate"].map(
            lambda x: owners_dict.get(x, "Unknown")
        )

        return result_df

    def get_analytics_metrics(self):
        total = self.plate_repo.get_total_count()
        last_ts = self.plate_repo.get_last_timestamp()
        last_seen = (
            datetime.fromtimestamp(last_ts).strftime("%Y-%m-%d %H:%M:%S")
            if last_ts else "—"
        )
        return total, last_seen

    def get_filtered_analytics(self, cameras, start, end):
        return self.plate_repo.get_analytics_data(cameras, start, end)

    def get_analytics_datasets(self):
        return {
            "camera_counts": self.plate_repo.get_counts_by_camera(),
            "vehicle_summary": self.plate_repo.get_vehicle_type_summary(),
            "timeline": self.plate_repo.get_hourly_timeline(),
            "vehicle_timeline": self.plate_repo.get_vehicle_type_timeline(),
        }

    def compute_durations(self, df):
        if df.empty:
            return df

        df["entry_time"] = pd.to_datetime(df["entry_time"], unit="s")
        df["exit_time"] = pd.to_datetime(df["exit_time"], unit="s")
        df["duration_sec"] = (
            df["exit_time"] - df["entry_time"]
        ).dt.total_seconds().abs()
        df["duration_hms"] = df["duration_sec"].apply(format_duration)
        return df

    def search_vehicles(self, filters):
        return self.plate_repo.search_logs(
            plate=filters.get("plate"),
            camera=filters.get("camera"),
            vehicle_type=filters.get("vehicle_type"),
            start=filters.get("start"),
            end=filters.get("end"),
        )

    def get_latest_vehicle_log_timestamp(self, filters):
        return self.plate_repo.get_latest_log_timestamp(
            plate=filters.get("plate"),
            camera=filters.get("camera"),
            vehicle_type=filters.get("vehicle_type"),
            plate_status=filters.get("plate_status"),
            start=filters.get("start"),
            end=filters.get("end"),
        )

    def get_latest_campus_event_timestamp(
            self,
            entry_cam,
            exit_cam,
            end,
            plate=None,
            vehicle_type=None,
    ):
        return self.plate_repo.get_latest_entry_exit_event_timestamp(
            entry_cam=entry_cam,
            exit_cam=exit_cam,
            end_datetime=end,
            plate=plate,
            vehicle_type=vehicle_type,
        )


# ==============================
# UTILITIES
# ==============================

def normalize_plate(value):
    return re.sub(r"[^A-Z0-9]", "", str(value or "").upper())


def format_duration(seconds):
    seconds = int(seconds)
    return f"{seconds//3600:02d}:{(seconds%3600)//60:02d}:{seconds%60:02d}"


# ==============================
# BASE PAGE
# ==============================

class BasePage(ABC):
    def __init__(self, title):
        self.title = title

    def show_title(self):
        st.subheader(self.title)

    @abstractmethod
    def render(self):
        pass


# ==============================
# UI PAGES
# ==============================

class LiveMonitorPage(BasePage):
    STATUS_STALE_AFTER_SECONDS = 10

    def __init__(
        self,
        plate_repo,
        live_frames_dir,
        stream_source_mode="frame_files",
        streams_api_url=None,
        streams_public_base_url=None,
    ):
        super().__init__("📡 Live Camera View")
        self.repo = plate_repo
        self.live_frames_dir = str(live_frames_dir)
        self.stream_source_mode = self._normalize_stream_source_mode(stream_source_mode)
        self.streams_api_url = str(streams_api_url or "").strip()
        self.streams_public_base_url = str(streams_public_base_url or "").strip()

    @staticmethod
    def _normalize_stream_source_mode(stream_source_mode):
        mode = str(stream_source_mode or "frame_files").strip().lower()
        aliases = {
            "live_frames_dir": "frame_files",
            "live_frames": "frame_files",
            "frame_files": "frame_files",
            "stream_api": "stream_api",
            "api": "stream_api",
        }
        return aliases.get(mode, "frame_files")

    def _is_stream_alive(self, payload):
        if not isinstance(payload, dict):
            return False

        status_value = str(payload.get("status") or "").strip().lower()
        if status_value != "running":
            return False

        last_update_text = str(payload.get("last_update") or "").strip()
        if not last_update_text:
            return False

        try:
            if "T" in last_update_text:
                last_update = datetime.fromisoformat(last_update_text.replace("Z", "+00:00"))
                current_time = datetime.now(last_update.tzinfo or timezone.utc)
            else:
                last_update = datetime.strptime(last_update_text, "%Y-%m-%d %H:%M:%S")
                current_time = datetime.now()
        except ValueError:
            return False

        age_seconds = (current_time - last_update).total_seconds()
        return age_seconds <= self.STATUS_STALE_AFTER_SECONDS

    def _streams_api_base_url(self):
        if not self.streams_api_url:
            return ""
        parts = urlsplit(self.streams_api_url)
        if not parts.scheme or not parts.netloc:
            return ""
        return f"{parts.scheme}://{parts.netloc}"

    def _streams_public_base(self):
        candidate = self.streams_public_base_url or self._streams_api_base_url()
        parts = urlsplit(candidate)
        if not parts.scheme or not parts.netloc:
            return ""
        return f"{parts.scheme}://{parts.netloc}"

    def _resolve_public_stream_url(self, video_url):
        video_url = str(video_url or "").strip()
        if not video_url:
            return ""

        public_base = self._streams_public_base()
        if not public_base:
            return video_url

        parsed = urlsplit(video_url)
        path = parsed.path or ""
        if parsed.query:
            path = f"{path}?{parsed.query}"

        if path:
            return urljoin(f"{public_base}/", path.lstrip("/"))
        return video_url

    def _load_stream_cards_from_files(self, selected_cameras):
        cards = []
        latest_frame_mtime = 0.0

        for cam in selected_cameras:
            image_path = os.path.join(self.live_frames_dir, f"{cam}.jpg")
            card = {
                "stream_name": str(cam),
                "frame_path": image_path,
                "alive": False,
                "mtime": None,
                "error": None,
                "source_mode": "frame_files",
            }

            if os.path.exists(image_path):
                try:
                    mtime = os.path.getmtime(image_path)
                    current_time = datetime.now().timestamp()
                    card["alive"] = (current_time - mtime) <= 30
                    card["mtime"] = mtime
                    latest_frame_mtime = max(latest_frame_mtime, mtime)
                except Exception as exc:
                    card["error"] = str(exc)

            cards.append(card)

        return cards, latest_frame_mtime

    def _load_stream_cards_from_api(self):
        if not self.streams_api_url:
            return []

        try:
            response = requests.get(self.streams_api_url, timeout=5)
            response.raise_for_status()
            data = response.json()
        except Exception:
            return []

        if isinstance(data, list):
            streams = data
        elif isinstance(data, dict):
            streams = data.get("streams", [])
        else:
            return []

        if not isinstance(streams, list):
            return []

        base_url = self._streams_api_base_url()
        cards = []
        for entry in streams:
            if not isinstance(entry, dict):
                continue

            payload = entry.get("status")
            if not isinstance(payload, dict):
                payload = {}

            stream_id = entry.get("stream_id")
            display_name = str(
                payload.get("display_name")
                or payload.get("source")
                or entry.get("source")
                or ""
            ).strip()
            cards.append(
                {
                    "stream_id": stream_id,
                    "stream_name": display_name or f"Stream {stream_id}",
                    "video_url": urljoin(base_url, str(entry.get("video_url") or "").strip()),
                    "status_url": urljoin(base_url, str(entry.get("status_url") or "").strip()),
                    "payload": payload,
                    "alive": self._is_stream_alive(payload),
                    "source_mode": "stream_api",
                }
            )

        return cards

    def _render_stream_api_card(self, card):
        payload = card.get("payload") or {}
        public_video_url = self._resolve_public_stream_url(card.get("video_url"))

        if public_video_url:
            components.html(
                f"""
                <div style="width:100%;background:#000;border-radius:0.5rem;overflow:hidden;">
                  <img
                    src="{public_video_url}"
                    alt="{card.get('stream_name', 'Live Stream')}"
                    style="display:block;width:100%;height:auto;min-height:260px;object-fit:contain;background:#000;"
                  />
                </div>
                """,
                height=320,
            )
        else:
            st.info("No live stream URL available.")

        metric_cols = st.columns(3)
        metric_cols[0].metric("Detected", int(payload.get("detected_count", 0) or 0))
        metric_cols[1].metric("Recognized", int(payload.get("recognized_count", 0) or 0))
        metric_cols[2].metric("FPS", f"{float(payload.get('fps', 0.0) or 0.0):.2f}")

        st.caption(
            f"Source: {payload.get('source', 'N/A')} | "
            f"Started: {payload.get('stream_started_at', 'N/A')} | "
            f"Last Update: {payload.get('last_update', 'N/A')}"
        )
        if card.get("status_url"):
            st.caption(f"Status URL: {card['status_url']}")

    def render(self):
        self.show_title()
        is_stream_api_mode = self.stream_source_mode == "stream_api"

        if is_stream_api_mode:
            st.caption("Live vehicle stream overview using URL stream API.")
            controls = st.columns([1, 1, 1, 2])
            auto_refresh = controls[0].toggle(
                "Auto refresh",
                value=False,
                key="vehicle_live_auto_refresh",
                help="Refresh live stream status periodically. Enabling this restarts MJPEG playback on each rerun.",
            )
            refresh_interval = controls[1].selectbox(
                "Interval",
                options=[5, 15, 30, 60],
                index=1,
                key="vehicle_live_refresh_interval",
                disabled=not auto_refresh,
            )
            controls[2].button("Refresh now", key="vehicle_live_refresh_now")
            if auto_refresh:
                st_autorefresh(
                    interval=int(refresh_interval) * 1000,
                    key="vehicle_live_status_refresh",
                )
            else:
                controls[3].caption("Auto refresh is off.")

            stream_cards = [card for card in self._load_stream_cards_from_api() if card["alive"]]
            if not stream_cards:
                st.info("No live vehicle streams were found.")
                return

            cols = st.columns(2)
            for idx, card in enumerate(stream_cards):
                with cols[idx % 2]:
                    with st.container(border=True):
                        st.subheader(card["stream_name"])
                        if card["alive"]:
                            st.success("stream is alive")
                        else:
                            st.error("stream is not alive")
                        self._render_stream_api_card(card)
            return

        cameras = self.repo.get_cameras()
        if not cameras:
            st.warning("No cameras detected yet.")
            return

        selected_cameras = st.multiselect("Select Cameras", cameras, default=cameras)
        st.caption("Live view auto-updates as soon as new frame files are detected.")
        if not selected_cameras:
            st.info("Select at least one camera.")
            return

        st.divider()
        stream_cards, latest_frame_mtime = self._load_stream_cards_from_files(selected_cameras)
        columns_per_row = 2 if len(selected_cameras) > 1 else 1

        for i in range(0, len(stream_cards), columns_per_row):
            row_cards = stream_cards[i:i + columns_per_row]
            cols = st.columns(columns_per_row)

            for col, card in zip(cols, row_cards):
                with col:
                    st.markdown(f"### 🎥 {card['stream_name']}")
                    image_path = card["frame_path"]

                    if os.path.exists(image_path):
                        if card["alive"]:
                            try:
                                img = Image.open(image_path)
                                img.verify()
                                img = Image.open(image_path)
                                st.image(img, width="content")

                                modified_time = datetime.fromtimestamp(card["mtime"]).strftime("%Y-%m-%d %H:%M:%S")
                                st.caption(f"Last updated: {modified_time}")
                                st.success("🟢 Stream is live")
                            except Exception:
                                st.info("🔄 Loading latest frame...")
                                st.empty()
                        else:
                            st.warning("🔴 Stream is not live")
                            if card["mtime"] is not None:
                                stale_time = datetime.fromtimestamp(card["mtime"]).strftime("%Y-%m-%d %H:%M:%S")
                                st.info(f"Last frame captured at: {stale_time}")
                    else:
                        st.info("⏳ Waiting for camera feed...")

        previous_mtime = st.session_state.get("live_last_frame_mtime", 0.0)
        st.session_state["live_last_frame_mtime"] = max(previous_mtime, latest_frame_mtime)
        st_autorefresh(interval=400, key="live_refresh_fast")


class AnalyticsPage(BasePage):
    def __init__(self, service):
        super().__init__("📊 Reports & Insights")
        self.service = service

    def render(self):
        self.show_title()

        cameras = self.service.plate_repo.get_cameras()

        view = st.segmented_control(
            "Select View",
            ["Reports & Insights", "Top Vehicles by Time Spent"],
            default="Reports & Insights",
        )

        if view == "Reports & Insights":
            # -------------------------
            # FILTERS
            # -------------------------
            col1, col2, col3 = st.columns(3)

            selected_cameras = col1.multiselect(
                "Filter by Camera",
                cameras,
                default=cameras
            )

            start_date = col2.date_input("Start Date")
            end_date = col3.date_input("End Date")

            total, last_ts, cam_df, timeline_df, vehicle_summary_df, vehicle_timeline_df = \
                self.service.get_filtered_analytics(
                    selected_cameras,
                    str(start_date) if start_date else None,
                    str(end_date) if end_date else None,
                )

            last_seen = (
                datetime.fromtimestamp(last_ts).strftime("%Y-%m-%d %H:%M:%S")
                if last_ts else "—"
            )

            # -------------------------
            # KPI METRICS
            # -------------------------
            m1, m2 = st.columns(2)
            m1.metric("Total Detections", total)
            m2.metric("Last Detection Time", last_seen)

            st.divider()

            # -------------------------
            # EXISTING ANALYTICS
            # -------------------------
            c1, c2 = st.columns(2)
            c3, c4 = st.columns(2)

            if not cam_df.empty:
                fig = px.pie(cam_df, values="count", names="camera_id", hole=0.5)
                fig.update_layout(title="Plates by Camera")
                c1.plotly_chart(fig, width='content')

            if not timeline_df.empty:
                timeline_df["bucket"] = pd.to_datetime(timeline_df["bucket"])
                fig = px.line(timeline_df, x="bucket", y="count", markers=True)
                fig.update_layout(title="Detection Timeline")
                c2.plotly_chart(fig, width='content')

            if not vehicle_timeline_df.empty:
                vehicle_timeline_df["bucket"] = pd.to_datetime(vehicle_timeline_df["bucket"])
                fig = px.bar(
                    vehicle_timeline_df,
                    x="bucket",
                    y="count",
                    color="vehicle_type",
                    barmode="group",
                )
                fig.update_layout(title="Vehicle Types Over Time")
                c3.plotly_chart(fig, width='content')

            if not vehicle_summary_df.empty:
                fig = px.bar(
                    vehicle_summary_df,
                    x="vehicle_type",
                    y="count",
                    color="vehicle_type",
                )
                fig.update_layout(title="Vehicle Type Summary")
                c4.plotly_chart(fig, width='content')

        if view == "Top Vehicles by Time Spent":
            # -------------------------
            # TOP TIME SPENT GRAPH
            # -------------------------
            st.subheader("⏱️ Top Vehicles by Time Spent in Campus")

            if len(cameras) < 2:
                st.warning("At least two cameras are required for time-spent analytics.")
                return

            t1, t2, t3, t4 = st.columns(4)

            entry_cam = t1.selectbox(
                "Entry Camera",
                cameras,
                index=0,
                key="analytics_entry"
            )

            exit_cam = t2.selectbox(
                "Exit Camera",
                cameras,
                index=1 if len(cameras) > 1 else 0,
                key="analytics_exit"
            )

            top_n = t3.selectbox("Top N", [5, 10, 20, 50], index=1)

            date_filter = t4.date_input("Date", key="analytics_date")

            if entry_cam == exit_cam:
                st.warning("Entry and Exit cameras must be different.")
                return

            if date_filter:
                # Safer full-day boundaries
                start_datetime = datetime.combine(
                    date_filter,
                    datetime.min.time()
                ).strftime("%Y-%m-%d %H:%M:%S")

                end_datetime = datetime.combine(
                    date_filter,
                    datetime.strptime("23:59:59", "%H:%M:%S").time()
                ).strftime("%Y-%m-%d %H:%M:%S")

                df_time = self.service.get_top_time_spent_data(
                    entry_cam,
                    exit_cam,
                    start_datetime,
                    end_datetime,
                )

                if not df_time.empty:

                    # Ensure sorted before slicing
                    df_time = df_time.sort_values(
                        "duration_minutes",
                        ascending=False
                    ).head(top_n)

                    # Convert minutes → HH:MM
                    df_time["duration_hm"] = df_time["duration_minutes"].apply(
                        lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}"
                    )

                    fig = px.bar(
                        df_time,
                        x="duration_minutes",
                        y="plate",
                        text="duration_hm",
                        color="duration_minutes",
                        orientation="h",
                    )

                    fig.update_traces(
                        textposition="outside",
                        hovertemplate=(
                            "<b>Plate:</b> %{y}<br>"
                            "<b>Time Spent:</b> %{text}<br>"
                            "<b>Minutes:</b> %{x:.2f}<extra></extra>"
                        ),
                    )

                    fig.update_layout(
                        title=f"Top {top_n} Vehicles by Time Spent",
                        xaxis_title="Time Spent (minutes, HH:MM shown on bars)",
                        yaxis_title="Plate",
                        height=520,
                    )
                    fig.update_yaxes(categoryorder="total ascending")

                    st.plotly_chart(fig, width='stretch')

                else:
                    st.info("No visit data available for selected date.")

class VehicleSearchPage(BasePage):
    def __init__(self, service):
        super().__init__("🔍 Search Vehicles")
        self.service = service

    def render(self):
        self.show_title()

        cameras = self.service.plate_repo.get_cameras()
        vehicle_types = self.service.plate_repo.get_vehicle_types()

        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)

        plate_filter = col1.text_input(
            "Plate contains",
            placeholder="e.g., LEA, ABC123",
        )

        camera_filter = col2.selectbox(
            "Camera",
            ["All"] + cameras,
        )

        vehicle_filter = col3.selectbox(
            "Vehicle Type",
            ["All"] + vehicle_types,
        )

        status_filter = col4.selectbox(
            "Plate Status",
            ["All", "Success", "Failed", "Partial"],
        )

        start_date = col5.date_input("Start Date")
        start_time = col6.time_input(
            "Start Time",
            value=time(0, 0)
        )

        end_date = col7.date_input("End Date")
        end_time = col8.time_input(
            "End Time",
            value=time(23, 45)
        )

        selected_start_dt = datetime.combine(
            start_date,
            start_time if start_time else datetime.min.time(),
        )
        selected_end_dt = datetime.combine(
            end_date,
            end_time if end_time else datetime.max.time(),
        )
        st.caption(
            "Selected Range: "
            f"{selected_start_dt.strftime('%Y-%m-%d %I:%M %p')} → "
            f"{selected_end_dt.strftime('%Y-%m-%d %I:%M %p')}"
        )
        vehicle_count_placeholder = st.empty()

        refresh_interval = 5

        start_ts = None
        end_ts = None

        if start_date:
            start_ts = int(
                datetime.combine(
                    start_date,
                    start_time if start_time else datetime.min.time()
                ).timestamp()
            )

        if end_date:
            end_ts = int(
                datetime.combine(
                    end_date,
                    end_time if end_time else datetime.max.time()
                ).timestamp()
            )

        filters = {
            "plate": plate_filter or None,
            "camera": None if camera_filter == "All" else camera_filter,
            "vehicle_type": None if vehicle_filter == "All" else vehicle_filter,
            "plate_status": None if status_filter == "All" else status_filter,
            "start": start_ts,
            "end": end_ts,
        }

        st_autorefresh(interval=refresh_interval * 1000, key="vehicle_search_refresh")

        filter_signature = (
            filters["plate"],
            filters["camera"],
            filters["vehicle_type"],
            filters["plate_status"],
            filters["start"],
            filters["end"],
        )

        last_signature = st.session_state.get("vehicle_search_last_signature")
        last_seen_ts = st.session_state.get("vehicle_search_last_seen_ts")
        cached_df = st.session_state.get("vehicle_search_cached_df")

        latest_ts = self.service.get_latest_vehicle_log_timestamp(filters)
        filters_changed = filter_signature != last_signature
        has_new_data = latest_ts != last_seen_ts
        cache_missing = cached_df is None

        if filters_changed or has_new_data or cache_missing:
            # Get total count with filters
            total_count = self.service.plate_repo.search_logs_count(
                plate=filters.get("plate"),
                camera=filters.get("camera"),
                vehicle_type=filters.get("vehicle_type"),
                plate_status=filters.get("plate_status"),
                start=filters.get("start"),
                end=filters.get("end"),
            )

            # Create paginator
            paginator = PaginationManager('anpr_search', total_count, default_per_page=50)

            # Load paginated data
            df = self.service.plate_repo.search_logs(
                page=paginator.current_page,
                per_page=paginator.per_page,
                plate=filters.get("plate"),
                camera=filters.get("camera"),
                vehicle_type=filters.get("vehicle_type"),
                plate_status=filters.get("plate_status"),
                start=filters.get("start"),
                end=filters.get("end"),
            )

            st.session_state["vehicle_search_cached_df"] = df
            st.session_state["vehicle_search_last_seen_ts"] = latest_ts
            st.session_state["vehicle_search_last_signature"] = filter_signature
        else:
            df = cached_df
            total_count = self.service.plate_repo.search_logs_count(
                plate=filters.get("plate"),
                camera=filters.get("camera"),
                vehicle_type=filters.get("vehicle_type"),
                plate_status=filters.get("plate_status"),
                start=filters.get("start"),
                end=filters.get("end"),
            )
            paginator = PaginationManager('anpr_search', total_count, default_per_page=50)

        if df.empty:
            st.warning("No matching records found.")
        else:
            # Helper function to format plate status with emoji
            def format_plate_status(status):
                if pd.isna(status) or status == "":
                    return "⚪ Unknown"
                status_lower = str(status).lower().strip()
                if status_lower == "success":
                    return "✅ Success"
                elif status_lower == "failed":
                    return "❌ Failed"
                elif status_lower == "partial":
                    return "⚠️ Partial"
                else:
                    return f"⚪ {str(status)}"
            
            header = st.columns([1.5, 1, 1.2, 0.8, 1, 1.5, 1.5, 1.5, 2, 2])
            header[0].markdown("**Time**")
            header[1].markdown("**Camera**")
            header[2].markdown("**Plate**")
            header[3].markdown("**Status**")
            header[4].markdown("**Vehicle Type**")
            header[5].markdown("**Owner Name**")
            header[6].markdown("**Failure Reason**")
            header[7].markdown("**Raw OCR**")
            header[8].markdown("**Plate Image**")
            header[9].markdown("**Vehicle Image**")

            with st.container(height=520):
                for _, row in df.iterrows():
                    cols = st.columns([1.5, 1, 1.2, 0.8, 1, 1.5, 1.5, 1.5, 2, 2])

                    cols[0].write(row["time"])
                    cols[1].write(row["camera_id"])
                    cols[2].write(row["plate"])
                    
                    # Plate Status with emoji indicator
                    cols[3].markdown(format_plate_status(row.get("plate_status", "")))
                    
                    cols[4].write(row["vehicle_type"])
                    cols[5].write(row.get("owner_name", "Unknown"))
                    
                    # Failure Reason - only show if status is failed or partial
                    failure_reason = row.get("failure_reason", "")
                    if pd.notna(failure_reason) and str(failure_reason).strip():
                        cols[6].warning(f"⚠️ {failure_reason}")
                    else:
                        cols[6].write("—")
                    
                    # Raw OCR Text - show in expandable section if available
                    raw_ocr = row.get("raw_ocr_text", "")
                    if pd.notna(raw_ocr) and str(raw_ocr).strip():
                        with cols[7]:
                            with st.expander("📝 View"):
                                st.code(str(raw_ocr), language=None)
                    else:
                        cols[7].write("—")

                    with cols[8]:
                        try:
                            image_path = row["plate_image"]

                            if image_path and isinstance(image_path, str) and os.path.exists(image_path):
                                st.image(image_path, width=130)
                            else:
                                st.write("—")

                        except Exception:
                            st.write("Invalid Image")

                    with cols[9]:
                        try:
                            image_path = row["vehicle_image"]

                            if image_path and isinstance(image_path, str) and os.path.exists(image_path):
                                st.image(image_path, width=180)
                            else:
                                st.write("—")

                        except Exception:
                            st.write("Invalid Image")

                    st.divider()

            # Pagination controls AFTER the table
            if paginator.render_pagination_controls():
                st.rerun()

class CampusTimeTrackerPage(BasePage):
    def __init__(self, service):
        super().__init__("⏱️ Time Spent on Campus")
        self.service = service

    def render(self):
        self.show_title()

        cameras = self.service.plate_repo.get_cameras()

        if len(cameras) < 2:
            st.warning("At least 2 cameras required.")
            return

        from datetime import time

        # ---- SESSION STATE INIT ----
        if "campus_start_time" not in st.session_state:
            st.session_state.campus_start_time = time(0, 0)

        if "campus_end_time" not in st.session_state:
            st.session_state.campus_end_time = time(23, 0)

        vehicle_types = self.service.plate_repo.get_vehicle_types()
        filters = st.columns([1, 1, 1, 1, 1, 1, 1, 1])

        entry_cam = filters[0].selectbox("Entry Camera", cameras, index=0)
        exit_cam = filters[1].selectbox("Exit Camera", cameras, index=1)
        plate_filter = filters[2].text_input(
            "Plate contains",
            placeholder="e.g., LEA, ABC123",
        )
        vehicle_type_filter = filters[3].selectbox(
            "Vehicle Type",
            ["All"] + vehicle_types,
        )

        start_date = filters[4].date_input("Start Date", datetime.now().date())

        start_time = filters[5].time_input(
            "Start Time",
            key="campus_start_time"
        )

        end_date = filters[6].date_input("End Date", datetime.now().date())

        end_time = filters[7].time_input(
            "End Time",
            key="campus_end_time"
        )

        # ---- Show selected date/time range ----
        selected_start_dt = datetime.combine(start_date, start_time)
        selected_end_dt = datetime.combine(end_date, end_time)
        range_count_line = st.empty()

        refresh_interval = 5

        if entry_cam == exit_cam:
            st.error("Entry and Exit cameras must be different.")
            return

        start_datetime = datetime.combine(start_date, start_time)
        end_datetime = datetime.combine(end_date, end_time)

        if end_datetime <= start_datetime:
            st.error("End date/time must be after start date/time.")
            return

        start_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
        end_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")
        plate_value = plate_filter or None
        vehicle_type_value = None if vehicle_type_filter == "All" else vehicle_type_filter

        campus_signature = (
            entry_cam,
            exit_cam,
            start_str,
            end_str,
            plate_value,
            vehicle_type_value,
        )
        previous_signature = st.session_state.get("campus_time_last_signature")
        previous_last_ts = st.session_state.get("campus_time_last_seen_ts")
        cached_df = st.session_state.get("campus_time_cached_df")

        latest_ts = self.service.get_latest_campus_event_timestamp(
            entry_cam=entry_cam,
            exit_cam=exit_cam,
            end=end_str,
            plate=plate_value,
            vehicle_type=vehicle_type_value,
        )

        filters_changed = campus_signature != previous_signature
        has_new_data = latest_ts != previous_last_ts
        cache_missing = cached_df is None

        if filters_changed or has_new_data or cache_missing:
            df = self.service.get_campus_time_data(
                entry_cam,
                exit_cam,
                start_str,
                end_str,
                plate_value,
                vehicle_type_value,
            )
            st.session_state["campus_time_cached_df"] = df
            st.session_state["campus_time_last_seen_ts"] = latest_ts
            st.session_state["campus_time_last_signature"] = campus_signature
        else:
            df = cached_df

        if df.empty:
            range_count_line.caption(
                "Selected Range: "
                f"{selected_start_dt.strftime('%Y-%m-%d %I:%M %p')} → "
                f"{selected_end_dt.strftime('%Y-%m-%d %I:%M %p')}  |  "
                "Vehicles Completed Visit: 0"
            )
            st.info("No vehicles found with both entry and exit records.")
            return

        range_count_line.caption(
            "Selected Range: "
            f"{selected_start_dt.strftime('%Y-%m-%d %I:%M %p')} → "
            f"{selected_end_dt.strftime('%Y-%m-%d %I:%M %p')}  |  "
            f"Vehicles Completed Visit: {len(df)}"
        )

        view = st.segmented_control(
            "Select View",
            ["Table View", "Graph View"],
            default="Table View",
            key="campus_view_toggle",
        )

        if view == "Table View":
            header = st.columns([1, 2, 2, 2, 1, 2, 2])
            header[0].markdown("**Plate**")
            header[1].markdown("**Owner**")
            header[2].markdown("**Entry Time**")
            header[3].markdown("**Exit Time**")
            header[4].markdown("**Duration**")
            header[5].markdown("**Entry Image**")
            header[6].markdown("**Exit Image**")

            with st.container(height=560):
                for _, row in df.iterrows():
                    cols = st.columns([1, 2, 2, 2, 1, 2, 2])

                    cols[0].write(row["plate"])
                    cols[1].write(row["owner_name"])
                    cols[2].write(row["entry_time"].strftime("%Y-%m-%d %I:%M %p"))
                    cols[3].write(row["exit_time"].strftime("%Y-%m-%d %I:%M %p"))
                    cols[4].write(row["duration_hms"])

                    with cols[5]:
                        if row["entry_image"] and os.path.exists(row["entry_image"]):
                            st.image(row["entry_image"], width=170)
                        else:
                            st.write("—")

                    with cols[6]:
                        if row["exit_image"] and os.path.exists(row["exit_image"]):
                            st.image(row["exit_image"], width=170)
                        else:
                            st.write("—")

                    st.divider()

        if view == "Graph View":
            st.subheader("📊 Time Spent Distribution")

            import plotly.express as px

            pie_df = (
                df.groupby("plate", as_index=False)["duration_minutes"]
                .sum()
                .sort_values("duration_minutes", ascending=False)
            )
            pie_df["duration_hm"] = pie_df["duration_minutes"].apply(
                lambda x: f"{int(x // 60):02d}:{int(x % 60):02d}"
            )

            fig = px.pie(
                pie_df,
                names="plate",
                values="duration_minutes",
                hole=0.25,
                title="🚗 Total Time Spent by Plate",
            )

            fig.update_traces(
                text=pie_df["duration_hm"],
                textposition="inside",
                textinfo="label+text",
                hovertemplate=(
                    "<b>Plate:</b> %{label}<br>"
                    "<b>Time Spent:</b> %{text}<br>"
                    "<b>Minutes:</b> %{value:.2f}<extra></extra>"
                ),
            )

            fig.update_layout(
                template="plotly_dark",
                title_font_size=24,
                height=760,
                showlegend=True,
                legend=dict(
                    orientation="v",
                    yanchor="middle",
                    y=0.5,
                    xanchor="left",
                    x=1.02,
                ),
                margin=dict(l=20, r=20, t=70, b=20),
            )

            left_spacer, center_col, right_spacer = st.columns([1, 8, 1])
            with center_col:
                st.plotly_chart(fig, width='stretch')


class RegisterVehiclePage(BasePage):
    def __init__(self, repo):
        super().__init__("📝 Manage Vehicle Registry")
        self.repo = repo

    def render(self):
        self.show_title()
        self.repo.ensure_table()

        action = st.segmented_control(
            "Select Screen",
            ["Create", "Read", "Update", "Delete"],
            default="Create",
            key="registry_crud_view",
        )

        if action == "Create":
            st.subheader("➕ Register Vehicle")
            col1, col2 = st.columns(2)
            plate = col1.text_input(
                "Plate Number",
                key="create_plate",
                placeholder="e.g., LEA1234",
            )
            owner = col2.text_input(
                "Owner Name",
                key="create_owner",
                placeholder="e.g., Ali Khan",
            )
            notes = st.text_area(
                "Notes",
                key="create_notes",
                placeholder="e.g., Faculty member / White Corolla",
            )

            if st.button("Register Vehicle", key="create_btn", width='stretch'):
                if not plate or not owner:
                    st.error("Plate and Owner are required.")
                elif self.repo.exists(plate):
                    st.warning("Vehicle already exists.")
                else:
                    self.repo.register(plate, owner, notes)
                    st.success("Vehicle registered successfully.")

        if action == "Read":
            st.subheader("🔎 Registered Vehicles")
            f1, f2 = st.columns(2)
            plate_filter = f1.text_input(
                "Search by Plate",
                key="read_plate_filter",
                placeholder="e.g., LEA",
            )
            owner_filter = f2.text_input(
                "Search by Owner",
                key="read_owner_filter",
                placeholder="e.g., Ahmad",
            )

            df = self.repo.search(
                plate_filter or None,
                owner_filter or None,
            )

            st.metric("Total Registered Vehicles", len(df))
            st.dataframe(df, width='stretch')

        if action == "Update":
            st.subheader("✏️ Update Vehicle")
            col1, col2 = st.columns(2)
            current_plate = col1.text_input(
                "Current Plate",
                key="update_current_plate",
                placeholder="e.g., LEA1234",
            )
            new_plate = col2.text_input(
                "New Plate (Optional)",
                key="update_new_plate",
                placeholder="e.g., LEA5678",
            )
            col3, col4 = st.columns(2)
            owner = col3.text_input(
                "Owner Name",
                key="update_owner",
                placeholder="e.g., Ali Khan",
            )
            notes = col4.text_input(
                "Notes",
                key="update_notes",
                placeholder="e.g., Updated department info",
            )

            if st.button("Update Vehicle", key="update_btn", width='stretch'):
                if not current_plate:
                    st.error("Current plate required.")
                elif not owner:
                    st.error("Owner required.")
                elif not self.repo.exists(current_plate):
                    st.warning("Vehicle not found.")
                else:
                    self.repo.update(current_plate, new_plate, owner, notes)
                    st.success("Vehicle updated successfully.")

        if action == "Delete":
            st.subheader("🗑️ Delete Vehicle")
            current_plate = st.text_input(
                "Current Plate",
                key="delete_current_plate",
                placeholder="e.g., LEA1234",
            )

            if st.button("Delete Vehicle", key="delete_btn", width='stretch'):
                if not current_plate:
                    st.error("Current plate required.")
                elif self.repo.delete(current_plate):
                    st.success("Vehicle deleted successfully.")
                else:
                    st.warning("Vehicle not found.")

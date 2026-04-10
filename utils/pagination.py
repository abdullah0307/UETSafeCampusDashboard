"""
Reusable pagination utility for Streamlit dashboard.
Provides server-side pagination with LIMIT/OFFSET pattern.
"""

import streamlit as st


class PaginationManager:
    """Manages pagination state and calculations."""
    
    def __init__(self, session_key: str, total_records: int, default_per_page: int = 50):
        """
        Initialize pagination manager.
        
        Args:
            session_key: Unique key for session state (e.g., 'lab_logs')
            total_records: Total number of records in the dataset
            default_per_page: Default records per page (25, 50, or 100)
        """
        self.session_key = session_key
        self.total_records = total_records
        self.default_per_page = default_per_page
        
        # Initialize session state
        self._init_session_state()
    
    def _init_session_state(self):
        """Initialize pagination session state if not exists."""
        page_key = f"{self.session_key}_page"
        per_page_key = f"{self.session_key}_per_page"
        
        if page_key not in st.session_state:
            st.session_state[page_key] = 1
        if per_page_key not in st.session_state:
            st.session_state[per_page_key] = self.default_per_page
    
    @property
    def current_page(self) -> int:
        """Get current page number."""
        return st.session_state.get(f"{self.session_key}_page", 1)
    
    @current_page.setter
    def current_page(self, value: int):
        """Set current page number."""
        st.session_state[f"{self.session_key}_page"] = max(1, min(value, self.total_pages))
    
    @property
    def per_page(self) -> int:
        """Get records per page."""
        return st.session_state.get(f"{self.session_key}_per_page", self.default_per_page)
    
    @per_page.setter
    def per_page(self, value: int):
        """Set records per page."""
        per_page_key = f"{self.session_key}_per_page"
        st.session_state[per_page_key] = value
        # Reset to page 1 when changing per_page
        st.session_state[f"{self.session_key}_page"] = 1
    
    @property
    def total_pages(self) -> int:
        """Calculate total number of pages."""
        if self.per_page <= 0:
            return 1
        return max(1, (self.total_records + self.per_page - 1) // self.per_page)
    
    @property
    def offset(self) -> int:
        """Calculate OFFSET for SQL query."""
        return (self.current_page - 1) * self.per_page
    
    @property
    def limit(self) -> int:
        """Get LIMIT for SQL query."""
        return self.per_page
    
    @property
    def start_record(self) -> int:
        """Get starting record number (1-indexed)."""
        if self.total_records == 0:
            return 0
        return self.offset + 1
    
    @property
    def end_record(self) -> int:
        """Get ending record number (1-indexed)."""
        return min(self.offset + self.per_page, self.total_records)
    
    @property
    def has_previous(self) -> bool:
        """Check if there's a previous page."""
        return self.current_page > 1
    
    @property
    def has_next(self) -> bool:
        """Check if there's a next page."""
        return self.current_page < self.total_pages
    
    def next_page(self):
        """Go to next page."""
        if self.has_next:
            self.current_page += 1
    
    def previous_page(self):
        """Go to previous page."""
        if self.has_previous:
            self.current_page -= 1
    
    def first_page(self):
        """Go to first page."""
        self.current_page = 1
    
    def last_page(self):
        """Go to last page."""
        self.current_page = self.total_pages
    
    def reset(self):
        """Reset pagination to first page."""
        st.session_state[f"{self.session_key}_page"] = 1
    
    def render_pagination_controls(self):
        """
        Render simple pagination UI controls with Next/Prev buttons.
        Returns True if state changed (needs rerun).
        """
        if self.total_records == 0:
            st.info("No records found.")
            return False
        
        state_changed = False
        
        # Display record range
        caption = f"📊 Showing records {self.start_record}-{self.end_record} of {self.total_records}"
        st.markdown(f"**{caption}**")
        
        # Controls row
        col_buttons, col_per_page = st.columns([4, 1])
        
        with col_buttons:
            btn_col1, btn_col2, btn_col3, btn_col4, btn_col5 = st.columns([1, 1, 2, 1, 1])
            
            with btn_col1:
                if st.button("⏮ First", key=f"{self.session_key}_first", disabled=not self.has_previous, use_container_width=True):
                    self.first_page()
                    state_changed = True
            
            with btn_col2:
                if st.button("◀ Prev", key=f"{self.session_key}_prev", disabled=not self.has_previous, use_container_width=True):
                    self.previous_page()
                    state_changed = True
            
            with btn_col3:
                st.markdown(
                    f"<div style='text-align:center; padding-top:8px;'>"
                    f"Page <strong>{self.current_page}</strong> of <strong>{self.total_pages}</strong>"
                    f"</div>",
                    unsafe_allow_html=True
                )
            
            with btn_col4:
                if st.button("Next ▶", key=f"{self.session_key}_next", disabled=not self.has_next, use_container_width=True):
                    self.next_page()
                    state_changed = True
            
            with btn_col5:
                if st.button("Last ⏭", key=f"{self.session_key}_last", disabled=not self.has_next, use_container_width=True):
                    self.last_page()
                    state_changed = True
        
        with col_per_page:
            new_per_page = st.selectbox(
                "Per page",
                options=[25, 50, 100],
                index=[25, 50, 100].index(self.per_page) if self.per_page in [25, 50, 100] else 1,
                key=f"{self.session_key}_per_page_select",
                label_visibility="collapsed"
            )
            if new_per_page != self.per_page:
                self.per_page = new_per_page
                state_changed = True
        
        return state_changed


def get_paginated_query(
    conn,
    table_name: str,
    page: int = 1,
    per_page: int = 50,
    columns: str = "*",
    where_clause: str = "",
    order_by: str = "id DESC",
    params: tuple = ()
):
    """
    Execute a paginated SQL query.
    
    Args:
        conn: SQLite connection
        table_name: Table name to query
        page: Page number (1-indexed)
        per_page: Records per page
        columns: Columns to select (default: "*")
        where_clause: WHERE clause without "WHERE" keyword
        order_by: ORDER BY clause
        params: Parameters for WHERE clause
    
    Returns:
        pandas DataFrame with paginated results
    """
    import pandas as pd
    
    offset = (page - 1) * per_page
    
    query = f"SELECT {columns} FROM {table_name}"
    if where_clause:
        query += f" WHERE {where_clause}"
    if order_by:
        query += f" ORDER BY {order_by}"
    query += f" LIMIT {per_page} OFFSET {offset}"
    
    return pd.read_sql_query(query, conn, params=params)


def get_total_count(
    conn,
    table_name: str,
    where_clause: str = "",
    params: tuple = ()
) -> int:
    """
    Get total count of records matching the criteria.
    
    Args:
        conn: SQLite connection
        table_name: Table name to query
        where_clause: WHERE clause without "WHERE" keyword
        params: Parameters for WHERE clause
    
    Returns:
        Total record count
    """
    query = f"SELECT COUNT(*) as total FROM {table_name}"
    if where_clause:
        query += f" WHERE {where_clause}"
    
    import pandas as pd
    result = pd.read_sql_query(query, conn, params=params)
    return int(result.iloc[0]['total'])

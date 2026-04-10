import streamlit as st
import streamlit.components.v1 as components


def clear_persisted_theme_once() -> None:
    """Clear browser-saved theme settings once per fresh tab load."""
    if st.session_state.get("_theme_reset_injected"):
        return

    st.session_state["_theme_reset_injected"] = True
    components.html(
        """
        <script>
        (function () {
          const marker = "uet-safe-campus-theme-reset";
          if (window.sessionStorage.getItem(marker)) {
            return;
          }

          window.sessionStorage.setItem(marker, "1");

          const stores = [window.localStorage, window.sessionStorage];
          for (const store of stores) {
            const keys = [];
            for (let i = 0; i < store.length; i += 1) {
              const key = store.key(i);
              if (!key) {
                continue;
              }

              if (key.toLowerCase().includes("theme")) {
                keys.push(key);
              }
            }

            for (const key of keys) {
              if (key !== marker) {
                store.removeItem(key);
              }
            }
          }
        })();
        </script>
        """,
        height=0,
        width=0,
    )

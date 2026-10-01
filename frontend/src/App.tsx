import { useEffect, useState } from "react";

import { apiBaseUrl } from "./lib/env";
import { checkApiHealth } from "./lib/api";
import { supabase } from "./lib/supabase";

type ApiStatus = "checking" | "online" | "offline";

export default function App() {
  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");

  useEffect(() => {
    const controller = new AbortController();

    checkApiHealth(controller.signal)
      .then(() => setApiStatus("online"))
      .catch(() => {
        if (!controller.signal.aborted) setApiStatus("offline");
      });

    return () => controller.abort();
  }, []);

  const statusLabel = {
    checking: "Checking API",
    online: "API online",
    offline: "API unavailable",
  }[apiStatus];

  return (
    <main className="page-shell">
      <header className="topbar">
        <a className="wordmark" href="/" aria-label="AI Document Intelligence home">
          <span className="wordmark-mark" aria-hidden="true">D</span>
          <span>DOCUMENT INTELLIGENCE</span>
        </a>
        <span className={`connection-status connection-status--${apiStatus}`}>
          <span className="status-dot" />
          {statusLabel}
        </span>
      </header>

      <section className="welcome" aria-labelledby="welcome-title">
        <p className="eyebrow">YOUR KNOWLEDGE WORKSPACE</p>
        <h1 id="welcome-title">Make your documents<br />work together.</h1>
        <p className="welcome-copy">
          Your workspace foundation is ready. Document tools will arrive in the next build phase.
        </p>
        <div className="foundation-list" aria-label="Foundation services">
          <div className="foundation-row">
            <span className="foundation-index">01</span>
            <span>Versioned API</span>
            <span className="foundation-value">{apiBaseUrl}</span>
          </div>
          <div className="foundation-row">
            <span className="foundation-index">02</span>
            <span>Supabase Auth client</span>
            <span className="foundation-value">{supabase ? "Configured" : "Not configured"}</span>
          </div>
          <div className="foundation-row">
            <span className="foundation-index">03</span>
            <span>Workspace foundation</span>
            <span className="foundation-value">Ready for Phase 1</span>
          </div>
        </div>
      </section>

      <footer className="page-footer">
        <span>AI DOCUMENT INTELLIGENCE</span>
        <span>FOUNDATION / 01</span>
      </footer>
    </main>
  );
}
import { useEffect, useState } from "react";
import type { FormEvent } from "react";

import { useAuth } from "./auth/AuthProvider";
import { checkApiHealth } from "./lib/api";
import { apiBaseUrl } from "./lib/env";
import { getWorkspaces } from "./workspaces/api";
import type { WorkspaceSummary } from "./workspaces/types";

type ApiStatus = "checking" | "online" | "offline";

export default function App() {
  const {
    status: authStatus,
    user,
    error: authError,
    signIn,
    signOut,
  } = useAuth();

  const [apiStatus, setApiStatus] = useState<ApiStatus>("checking");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [isSigningIn, setIsSigningIn] = useState(false);
  const [signInError, setSignInError] = useState<string | null>(null);
  const [isSigningOut, setIsSigningOut] = useState(false);

  const [workspaces, setWorkspaces] = useState<WorkspaceSummary[]>([]);
  const [isLoadingWorkspaces, setIsLoadingWorkspaces] = useState(false);
  const [workspaceError, setWorkspaceError] = useState<string | null>(null);
  const [selectedWorkspaceId, setSelectedWorkspaceId] = useState<string | null>(
    null,
  );

  useEffect(() => {
    const controller = new AbortController();

    checkApiHealth(controller.signal)
      .then(() => setApiStatus("online"))
      .catch(() => {
        if (!controller.signal.aborted) {
          setApiStatus("offline");
        }
      });

    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (authStatus !== "authenticated") {
      setWorkspaces([]);
      setSelectedWorkspaceId(null);
      setWorkspaceError(null);
      return;
    }

    const controller = new AbortController();

    async function loadWorkspaces() {
      setIsLoadingWorkspaces(true);
      setWorkspaceError(null);

      try {
        const result = await getWorkspaces(controller.signal);

        if (controller.signal.aborted) {
          return;
        }

        setWorkspaces(result);

        setSelectedWorkspaceId((currentId) => {
          if (currentId && result.some((workspace) => workspace.id === currentId)) {
            return currentId;
          }

          return result[0]?.id ?? null;
        });
      } catch (error) {
        if (controller.signal.aborted) {
          return;
        }

        setWorkspaceError(
          error instanceof Error
            ? error.message
            : "Unable to load workspaces.",
        );
      } finally {
        if (!controller.signal.aborted) {
          setIsLoadingWorkspaces(false);
        }
      }
    }

    void loadWorkspaces();

    return () => controller.abort();
  }, [authStatus, user?.id]);

  const handleSignIn = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();

    setSignInError(null);
    setIsSigningIn(true);

    try {
      await signIn(email.trim(), password);
    } catch {
      setSignInError(
        "Unable to sign in. Please check your email and password.",
      );
    } finally {
      setIsSigningIn(false);
    }
  };

  const handleSignOut = async () => {
    setIsSigningOut(true);

    try {
      await signOut();
    } catch {
      // AuthProvider retains the detailed error state.
    } finally {
      setIsSigningOut(false);
    }
  };

  if (authStatus === "loading") {
    return (
      <main className="page-shell">
        <section className="welcome" aria-live="polite">
          <p className="eyebrow">YOUR KNOWLEDGE WORKSPACE</p>
          <h1>
            Checking your
            <br />
            session.
          </h1>
          <p className="welcome-copy">
            Connecting your secure workspace.
          </p>
        </section>
      </main>
    );
  }

  if (authStatus === "configuration-error") {
    return (
      <main className="page-shell">
        <section className="welcome" aria-labelledby="configuration-title">
          <p className="eyebrow">CONFIGURATION REQUIRED</p>
          <h1 id="configuration-title">
            Authentication
            <br />
            isn't ready.
          </h1>
          <p className="welcome-copy">
            Supabase authentication is not configured for this frontend.
          </p>
        </section>
      </main>
    );
  }

  if (authStatus === "unauthenticated") {
    return (
      <main className="page-shell">
        <header className="topbar">
          <a className="wordmark" href="/" aria-label="AI Document Intelligence home">
            <span className="wordmark-mark" aria-hidden="true">D</span>
            <span>DOCUMENT INTELLIGENCE</span>
          </a>

          <span className={`connection-status connection-status--${apiStatus}`}>
            <span className="status-dot" />
            {apiStatus === "checking"
              ? "Checking API"
              : apiStatus === "online"
                ? "API online"
                : "API unavailable"}
          </span>
        </header>

        <section className="welcome" aria-labelledby="signin-title">
          <p className="eyebrow">SECURE WORKSPACE</p>
          <h1 id="signin-title">
            Welcome
            <br />
            back.
          </h1>
          <p className="welcome-copy">
            Sign in to access your documents, conversations, and
            AI-powered analysis.
          </p>

          <form onSubmit={handleSignIn} noValidate>
            <div>
              <label htmlFor="email">Email</label>
              <input
                id="email"
                name="email"
                type="email"
                autoComplete="email"
                value={email}
                onChange={(event) => setEmail(event.target.value)}
                required
              />
            </div>

            <div>
              <label htmlFor="password">Password</label>
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
              />
            </div>

            {(signInError || authError) && (
              <p role="alert">{signInError ?? authError}</p>
            )}

            <button type="submit" disabled={isSigningIn}>
              {isSigningIn ? "Signing in..." : "Sign in"}
            </button>
          </form>
        </section>

        <footer className="page-footer">
          <span>AI DOCUMENT INTELLIGENCE</span>
          <span>SECURE ACCESS</span>
        </footer>
      </main>
    );
  }

  const selectedWorkspace = workspaces.find(
    (workspace) => workspace.id === selectedWorkspaceId,
  );

  return (
    <main className="page-shell">
      <header className="topbar">
        <a className="wordmark" href="/" aria-label="AI Document Intelligence home">
          <span className="wordmark-mark" aria-hidden="true">D</span>
          <span>DOCUMENT INTELLIGENCE</span>
        </a>

        <div>
          <span className={`connection-status connection-status--${apiStatus}`}>
            <span className="status-dot" />
            {apiStatus === "checking"
              ? "Checking API"
              : apiStatus === "online"
                ? "API online"
                : "API unavailable"}
          </span>

          <p>{user?.email}</p>

          <button
            type="button"
            onClick={handleSignOut}
            disabled={isSigningOut}
          >
            {isSigningOut ? "Signing out..." : "Sign out"}
          </button>
        </div>
      </header>

      <section className="welcome" aria-labelledby="welcome-title">
        <p className="eyebrow">YOUR KNOWLEDGE WORKSPACE</p>

        <h1 id="welcome-title">
          Make your documents
          <br />
          work together.
        </h1>

        <p className="welcome-copy">
          Choose a workspace to prepare for your documents, conversations,
          and AI-powered analysis.
        </p>

        <section aria-labelledby="workspaces-title">
          <h2 id="workspaces-title">Your workspaces</h2>

          {isLoadingWorkspaces && (
            <p role="status">Loading your workspaces...</p>
          )}

          {workspaceError && (
            <div role="alert">
              <p>We couldn't load your workspaces.</p>
              <p>{workspaceError}</p>
              <button
                type="button"
                onClick={() => {
                  setWorkspaceError(null);
                  setIsLoadingWorkspaces(true);
                  void getWorkspaces()
                    .then((result) => {
                      setWorkspaces(result);
                      setSelectedWorkspaceId(result[0]?.id ?? null);
                    })
                    .catch((error: unknown) => {
                      setWorkspaceError(
                        error instanceof Error
                          ? error.message
                          : "Unable to load workspaces.",
                      );
                    })
                    .finally(() => setIsLoadingWorkspaces(false));
                }}
              >
                Try again
              </button>
            </div>
          )}

          {!isLoadingWorkspaces && !workspaceError && workspaces.length === 0 && (
            <p>
              You don't belong to a workspace yet. Ask a workspace owner or
              administrator to add you.
            </p>
          )}

          {!isLoadingWorkspaces && !workspaceError && workspaces.length > 0 && (
            <div className="foundation-list">
              {workspaces.map((workspace) => (
                <button
                  className="foundation-row"
                  key={workspace.id}
                  type="button"
                  aria-pressed={selectedWorkspaceId === workspace.id}
                  onClick={() => setSelectedWorkspaceId(workspace.id)}
                >
                  <span className="foundation-index">WS</span>
                  <span>{workspace.name}</span>
                  <span className="foundation-value">
                    {workspace.role}
                    {selectedWorkspaceId === workspace.id ? " · Selected" : ""}
                  </span>
                </button>
              ))}
            </div>
          )}

          {selectedWorkspace && (
            <p role="status">
              Selected workspace: <strong>{selectedWorkspace.name}</strong>
            </p>
          )}
        </section>

        <div className="foundation-list" aria-label="Foundation services">
          <div className="foundation-row">
            <span className="foundation-index">01</span>
            <span>Versioned API</span>
            <span className="foundation-value">{apiBaseUrl}</span>
          </div>

          <div className="foundation-row">
            <span className="foundation-index">02</span>
            <span>Supabase Auth</span>
            <span className="foundation-value">Authenticated</span>
          </div>

          <div className="foundation-row">
            <span className="foundation-index">03</span>
            <span>Workspace discovery</span>
            <span className="foundation-value">Connected</span>
          </div>
        </div>
      </section>

      <footer className="page-footer">
        <span>AI DOCUMENT INTELLIGENCE</span>
        <span>AUTHENTICATED / WORKSPACES</span>
      </footer>
    </main>
  );
}

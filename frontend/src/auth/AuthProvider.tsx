import {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";
import type { Session, User } from "@supabase/supabase-js";

import { supabase } from "../lib/supabase";

type AuthStatus =
  | "loading"
  | "authenticated"
  | "unauthenticated"
  | "configuration-error";

type AuthContextValue = {
  status: AuthStatus;
  session: Session | null;
  user: User | null;
  error: string | null;
  signIn: (email: string, password: string) => Promise<void>;
  signOut: () => Promise<void>;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

type AuthProviderProps = {
  children: ReactNode;
};

export function AuthProvider({ children }: AuthProviderProps) {
  const [status, setStatus] = useState<AuthStatus>(
    supabase ? "loading" : "configuration-error",
  );
  const [session, setSession] = useState<Session | null>(null);
  const [error, setError] = useState<string | null>(
    supabase ? null : "Supabase authentication is not configured.",
  );

  useEffect(() => {
    if (!supabase) {
      return;
    }

    let mounted = true;

    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, nextSession) => {
      if (!mounted) {
        return;
      }

      setSession(nextSession);
      setError(null);
      setStatus(nextSession ? "authenticated" : "unauthenticated");
    });

    supabase.auth
      .getSession()
      .then(({ data, error: sessionError }) => {
        if (!mounted) {
          return;
        }

        if (sessionError) {
          setSession(null);
          setError(sessionError.message);
          setStatus("unauthenticated");
          return;
        }

        setSession(data.session);
        setStatus(
          data.session ? "authenticated" : "unauthenticated",
        );
      })
      .catch(() => {
        if (!mounted) {
          return;
        }

        setSession(null);
        setError("Unable to initialize authentication.");
        setStatus("unauthenticated");
      });

    return () => {
      mounted = false;
      subscription.unsubscribe();
    };
  }, []);

  const signIn = async (email: string, password: string) => {
    if (!supabase) {
      setStatus("configuration-error");
      setError("Supabase authentication is not configured.");
      return;
    }

    setError(null);

    const { data, error: signInError } = await supabase.auth.signInWithPassword({
      email,
      password,
    });

    if (signInError) {
      setError(signInError.message);
      throw signInError;
    }

    setSession(data.session);
    setStatus(data.session ? "authenticated" : "unauthenticated");
  };

  const signOut = async () => {
    if (!supabase) {
      setStatus("configuration-error");
      setError("Supabase authentication is not configured.");
      return;
    }

    const { error: signOutError } = await supabase.auth.signOut();

    if (signOutError) {
      setError(signOutError.message);
      throw signOutError;
    }

    setSession(null);
    setStatus("unauthenticated");
    setError(null);
  };

  const value = useMemo<AuthContextValue>(
    () => ({
      status,
      session,
      user: session?.user ?? null,
      error,
      signIn,
      signOut,
    }),
    [status, session, error],
  );

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth(): AuthContextValue {
  const context = useContext(AuthContext);

  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider.");
  }

  return context;
}
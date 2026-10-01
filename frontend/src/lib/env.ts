const configuredApiUrl = import.meta.env.VITE_API_BASE_URL?.trim();

export const apiBaseUrl = (configuredApiUrl || "http://localhost:8000/api/v1").replace(/\/$/, "");

export const supabaseUrl = import.meta.env.VITE_SUPABASE_URL?.trim() || "";
export const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY?.trim() || "";
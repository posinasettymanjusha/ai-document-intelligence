import { createClient, type SupabaseClient } from "@supabase/supabase-js";

import { supabaseAnonKey, supabaseUrl } from "./env";

export const supabase: SupabaseClient | null =
  supabaseUrl && supabaseAnonKey ? createClient(supabaseUrl, supabaseAnonKey) : null;
export type ProviderName = "opencode" | "deepseek" | "local";

export interface CharacterProfile {
  line?: string;
  tags?: string[];
  scene?: string;
  about?: string;
}

export interface Character {
  id: string;
  name: string;
  tag: string;
  color: string;
  description: string;
  greeting: string;
  starters: string[];
  profile?: CharacterProfile;
  custom?: boolean;
  avatar_url?: string | null;
  gender?: string;
  identity?: Record<string, unknown>;
}

export interface ImageAttachment {
  id: string;
  mime_type: string;
  width: number;
  height: number;
}

export interface RoleplayDisplay {
  version: number;
  speech: string;
  inner_thought: string;
  action: string;
  state?: {
    mood: string;
    affection: number;
    relationship: string;
  };
}

export interface Message {
  id: number;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  image?: ImageAttachment;
  display?: RoleplayDisplay | null;
}

export interface Session {
  id: string;
  character_id: string;
  provider: ProviderName;
  model_id: string;
  model_mode: "base" | "lora";
  messages: Message[];
  created_at: string;
  updated_at: string;
}

export interface ProviderSettings {
  default_provider: ProviderName;
  deepseek: {
    model_id: string;
    configured: boolean;
    supports_text: boolean;
    supports_images: boolean;
    usage: Usage;
  };
  opencode: {
    model_id: string;
    configured: boolean;
    supports_text: boolean;
    supports_images: boolean;
    usage: Usage;
  };
  local: {
    available: boolean;
    model_id: string | null;
    supports_text: boolean;
    supports_images: boolean;
  };
  token_limits: {
    request_token_limit: number;
    daily_token_limit: number;
  };
}

export interface Usage {
  prompt_tokens: number;
  completion_tokens: number;
  cached_tokens: number;
  total_tokens: number;
  calls: number;
  usage_day: string;
}

export interface Health {
  status: "ready" | "model_unavailable";
  app_version?: string;
  default_provider: ProviderName;
  model_id: string | null;
  providers: Record<string, {
    available: boolean;
    configured?: boolean;
    model_id: string | null;
    supports_images: boolean;
  }>;
  token_limits: { request: number; daily: number };
  deepseek_usage: Usage;
  opencode_usage: Usage;
  structured_memory_enabled?: boolean;
  structured_memory_mode?: "automatic" | "manual" | "disabled";
  structured_memory_jobs?: Record<string, number>;
  memory_daily_token_limit?: number;
  speech_provider: string | null;
  asr_provider: string | null;
}

export interface MemoryItem {
  message_id: number;
  role: "user" | "assistant";
  quote: string;
}

export interface SessionMemory {
  items: MemoryItem[];
  covered_message_id: number;
  notes: string;
  auto_enabled: boolean;
  revision: number;
  excluded_message_ids?: number[];
}

export interface StructuredMemoryEvidence {
  message_id: number;
  role: "user" | "assistant";
  quote: string;
}

export interface StructuredMemoryRecord {
  id: number;
  character_id: string;
  subject: "user" | "character" | "shared";
  fact_key: string | null;
  value: string | null;
  content: string;
  kind: "user_report" | "character_detail" | "shared_event" | "belief" | "plan";
  evidence: StructuredMemoryEvidence[];
  event_date: string | null;
  observed_order: number;
  semantic_verified: boolean;
  state: "candidate" | "active" | "contested" | "superseded" | "deleted";
  supersedes: number | null;
  contested_with: number | null;
  auto_action: string | null;
  auto_reason: string | null;
  origin_job_id: number | null;
  created_at: string;
  updated_at: string;
}

export interface StructuredMemoryList {
  character_id: string;
  records: StructuredMemoryRecord[];
  counts: { current: number; candidate: number; contested: number };
  jobs?: Record<string, number>;
  usage?: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    calls: number;
    usage_day: string;
  };
  processed_through?: number;
}

export interface StructuredMemoryStatus {
  character_id: string;
  worker_running: boolean;
  jobs: Record<string, number>;
  usage_today: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    calls: number;
    usage_day: string;
  };
  daily_token_limit: number;
  automatic: boolean;
}

export interface CharacterCard {
  id: string;
  name: string;
  tagline: string;
  description: string;
  background: string;
  personality: string;
  speech_style: string;
  relationship: string;
  greeting: string;
  scenario: string;
  motivation: string;
  boundaries: string;
  source_notes: string;
  knowledge_boundary: {
    character_kind: "original" | "historical" | "fictional_canon";
    era: string;
    knowledge_cutoff: string;
    universe: string;
    allowed_knowledge: string;
    forbidden_knowledge: string;
    forbidden_terms: string[];
    can_learn_from_user: boolean;
    allow_everyday_invention: boolean;
  };
  fixed_facts: Array<{ key: string; value: string; status: "confirmed" | "proposal" }>;
  worldbook: Array<{
    id?: string;
    title: string;
    keywords: string[];
    content: string;
    priority?: number;
    enabled?: boolean;
  }>;
  example_dialogue: Array<{ user: string; character: string }>;
  generation_preset?: Record<string, number>;
  draft_notice?: string;
  schema_version?: number;
}

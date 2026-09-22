-- Run in the Supabase SQL Editor before starting the app.
create table if not exists public.chat_messages (
  id bigint generated always as identity primary key,
  user_id uuid not null references auth.users(id) on delete cascade,
  question text not null,
  answer text not null,
  sources jsonb not null default '[]'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists chat_messages_user_created_idx
  on public.chat_messages (user_id, created_at);

alter table public.chat_messages enable row level security;

create policy "Users can read their own chat messages"
  on public.chat_messages for select to authenticated
  using ((select auth.uid()) = user_id);

create policy "Users can create their own chat messages"
  on public.chat_messages for insert to authenticated
  with check ((select auth.uid()) = user_id);

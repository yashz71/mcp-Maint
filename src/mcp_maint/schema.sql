CREATE TABLE IF NOT EXISTS maintenance_tickets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    title TEXT NOT NULL,
    description TEXT NOT NULL,

    status TEXT NOT NULL DEFAULT 'draft'
        CHECK (status IN (
            'draft',
            'open',
            'in_progress',
            'resolved',
            'closed'
        )),

    equipment_id TEXT,
    evidence JSONB DEFAULT '[]'::jsonb,

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    approved_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_maintenance_tickets_status
ON maintenance_tickets(status);

CREATE INDEX IF NOT EXISTS idx_maintenance_tickets_created_at
ON maintenance_tickets(created_at DESC);
-- TallyIO Data Storage - TimescaleDB Hypertables
-- Production-ready schema for financial MEV/DeFi data

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Opportunities table (hypertable for time-series data)
CREATE TABLE IF NOT EXISTS opportunities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    opportunity_type VARCHAR(50) NOT NULL,
    chain_id INTEGER NOT NULL,
    protocol VARCHAR(100),
    token_in VARCHAR(42) NOT NULL,  -- Ethereum address format
    token_out VARCHAR(42) NOT NULL, -- Ethereum address format
    amount_in DECIMAL(78,0) NOT NULL, -- Support for 256-bit integers
    amount_out DECIMAL(78,0) NOT NULL,
    profit_eth DECIMAL(36,18) NOT NULL,
    gas_cost DECIMAL(36,18) NOT NULL,
    net_profit DECIMAL(36,18) NOT NULL,
    confidence_score DOUBLE PRECISION NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    executed BOOLEAN DEFAULT FALSE,
    execution_tx_hash VARCHAR(66), -- Ethereum transaction hash
    block_number BIGINT,
    transaction_index INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ,
    data JSONB,
    
    -- Constraints for data integrity
    CONSTRAINT valid_amounts CHECK (amount_in > 0 AND amount_out > 0),
    CONSTRAINT valid_profit CHECK (profit_eth >= 0),
    CONSTRAINT valid_gas_cost CHECK (gas_cost >= 0),
    CONSTRAINT valid_chain_id CHECK (chain_id > 0),
    CONSTRAINT valid_addresses CHECK (
        token_in ~ '^0x[a-fA-F0-9]{40}$' AND 
        token_out ~ '^0x[a-fA-F0-9]{40}$'
    ),
    CONSTRAINT valid_tx_hash CHECK (
        execution_tx_hash IS NULL OR 
        execution_tx_hash ~ '^0x[a-fA-F0-9]{64}$'
    )
);

-- Convert to hypertable (partitioned by time)
SELECT create_hypertable('opportunities', 'created_at', if_not_exists => TRUE);

-- Transactions table (hypertable for blockchain transactions)
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tx_hash VARCHAR(66) NOT NULL UNIQUE,
    chain_id INTEGER NOT NULL,
    block_number BIGINT NOT NULL,
    transaction_index INTEGER NOT NULL,
    from_address VARCHAR(42) NOT NULL,
    to_address VARCHAR(42),
    value_wei DECIMAL(78,0) NOT NULL DEFAULT 0,
    gas_limit BIGINT NOT NULL,
    gas_used BIGINT,
    gas_price DECIMAL(78,0) NOT NULL,
    nonce BIGINT NOT NULL,
    status INTEGER, -- 0 = failed, 1 = success
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data JSONB,
    
    -- Constraints
    CONSTRAINT valid_tx_hash CHECK (tx_hash ~ '^0x[a-fA-F0-9]{64}$'),
    CONSTRAINT valid_addresses_tx CHECK (
        from_address ~ '^0x[a-fA-F0-9]{40}$' AND
        (to_address IS NULL OR to_address ~ '^0x[a-fA-F0-9]{40}$')
    ),
    CONSTRAINT valid_gas CHECK (gas_limit > 0 AND gas_price >= 0),
    CONSTRAINT valid_status CHECK (status IS NULL OR status IN (0, 1))
);

-- Convert to hypertable
SELECT create_hypertable('transactions', 'created_at', if_not_exists => TRUE);

-- Blocks table (hypertable for blockchain blocks)
CREATE TABLE IF NOT EXISTS blocks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    block_number BIGINT NOT NULL,
    chain_id INTEGER NOT NULL,
    block_hash VARCHAR(66) NOT NULL,
    parent_hash VARCHAR(66) NOT NULL,
    timestamp_unix BIGINT NOT NULL,
    gas_limit BIGINT NOT NULL,
    gas_used BIGINT NOT NULL,
    base_fee_per_gas DECIMAL(78,0),
    difficulty DECIMAL(78,0),
    total_difficulty DECIMAL(78,0),
    transaction_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data JSONB,
    
    -- Unique constraint per chain
    UNIQUE(chain_id, block_number),
    
    -- Constraints
    CONSTRAINT valid_block_hash CHECK (block_hash ~ '^0x[a-fA-F0-9]{64}$'),
    CONSTRAINT valid_parent_hash CHECK (parent_hash ~ '^0x[a-fA-F0-9]{64}$'),
    CONSTRAINT valid_gas_block CHECK (gas_limit > 0 AND gas_used >= 0),
    CONSTRAINT valid_transaction_count CHECK (transaction_count >= 0)
);

-- Convert to hypertable
SELECT create_hypertable('blocks', 'created_at', if_not_exists => TRUE);

-- Events table (hypertable for smart contract events)
CREATE TABLE IF NOT EXISTS events (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chain_id INTEGER NOT NULL,
    block_number BIGINT NOT NULL,
    transaction_hash VARCHAR(66) NOT NULL,
    log_index INTEGER NOT NULL,
    contract_address VARCHAR(42) NOT NULL,
    event_signature VARCHAR(66) NOT NULL, -- Keccak256 hash of event signature
    topic0 VARCHAR(66),
    topic1 VARCHAR(66),
    topic2 VARCHAR(66),
    topic3 VARCHAR(66),
    data_hex TEXT,
    decoded_data JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT valid_event_addresses CHECK (contract_address ~ '^0x[a-fA-F0-9]{40}$'),
    CONSTRAINT valid_event_signature CHECK (event_signature ~ '^0x[a-fA-F0-9]{64}$'),
    CONSTRAINT valid_topics CHECK (
        (topic0 IS NULL OR topic0 ~ '^0x[a-fA-F0-9]{64}$') AND
        (topic1 IS NULL OR topic1 ~ '^0x[a-fA-F0-9]{64}$') AND
        (topic2 IS NULL OR topic2 ~ '^0x[a-fA-F0-9]{64}$') AND
        (topic3 IS NULL OR topic3 ~ '^0x[a-fA-F0-9]{64}$')
    )
);

-- Convert to hypertable
SELECT create_hypertable('events', 'created_at', if_not_exists => TRUE);

-- Performance indexes for opportunities
CREATE INDEX IF NOT EXISTS idx_opportunities_type_chain ON opportunities(opportunity_type, chain_id);
CREATE INDEX IF NOT EXISTS idx_opportunities_profit ON opportunities(profit_eth DESC) WHERE executed = FALSE;
CREATE INDEX IF NOT EXISTS idx_opportunities_created ON opportunities(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_opportunities_executed ON opportunities(executed, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_opportunities_protocol ON opportunities(protocol, chain_id);
CREATE INDEX IF NOT EXISTS idx_opportunities_tokens ON opportunities(token_in, token_out);
CREATE INDEX IF NOT EXISTS idx_opportunities_confidence ON opportunities(confidence_score DESC) WHERE executed = FALSE;

-- Performance indexes for transactions
CREATE INDEX IF NOT EXISTS idx_transactions_hash ON transactions(tx_hash);
CREATE INDEX IF NOT EXISTS idx_transactions_block ON transactions(chain_id, block_number);
CREATE INDEX IF NOT EXISTS idx_transactions_from ON transactions(from_address);
CREATE INDEX IF NOT EXISTS idx_transactions_to ON transactions(to_address) WHERE to_address IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_transactions_status ON transactions(status, created_at DESC);

-- Performance indexes for blocks
CREATE INDEX IF NOT EXISTS idx_blocks_number ON blocks(chain_id, block_number DESC);
CREATE INDEX IF NOT EXISTS idx_blocks_hash ON blocks(block_hash);
CREATE INDEX IF NOT EXISTS idx_blocks_timestamp ON blocks(timestamp_unix DESC);

-- Performance indexes for events
CREATE INDEX IF NOT EXISTS idx_events_contract ON events(contract_address, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_signature ON events(event_signature, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_events_block ON events(chain_id, block_number);
CREATE INDEX IF NOT EXISTS idx_events_tx ON events(transaction_hash);
CREATE INDEX IF NOT EXISTS idx_events_topics ON events(topic0, topic1) WHERE topic0 IS NOT NULL;

-- Compression policies for older data (TimescaleDB feature)
SELECT add_compression_policy('opportunities', INTERVAL '7 days');
SELECT add_compression_policy('transactions', INTERVAL '30 days');
SELECT add_compression_policy('blocks', INTERVAL '30 days');
SELECT add_compression_policy('events', INTERVAL '7 days');

-- Data retention policies (optional - for production environments)
-- SELECT add_retention_policy('opportunities', INTERVAL '1 year');
-- SELECT add_retention_policy('transactions', INTERVAL '2 years');
-- SELECT add_retention_policy('blocks', INTERVAL '2 years');
-- SELECT add_retention_policy('events', INTERVAL '1 year');

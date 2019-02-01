SET statement_timeout = 0;
SET lock_timeout = 0;
SET idle_in_transaction_session_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: 
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: 
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET search_path = public, pg_catalog;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: angle; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE angle (
    model character varying(30),
    phi double precision,
    psi double precision,
    omega double precision,
    resorder integer,
    id integer NOT NULL
);


ALTER TABLE angle OWNER TO postgres;

--
-- Name: angle_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE angle_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE angle_id_seq OWNER TO postgres;

--
-- Name: angle_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE angle_id_seq OWNED BY angle.id;


--
-- Name: atom; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE atom (
    model character(30),
    serial integer,
    name character varying(4),
    altloc character varying(1),
    resname character varying(3),
    chainid character varying(1),
    resseq integer,
    icode character varying(1),
    x double precision,
    y double precision,
    z double precision,
    occupancy double precision,
    tempfactor double precision,
    element character varying(2),
    id integer NOT NULL
);


ALTER TABLE atom OWNER TO postgres;

--
-- Name: atom_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE atom_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE atom_id_seq OWNER TO postgres;

--
-- Name: atom_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE atom_id_seq OWNED BY atom.id;


--
-- Name: model; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE model (
    code character(30) NOT NULL,
    filename character varying(40) NOT NULL,
    rvalue double precision,
    rfree double precision,
    resolution double precision,
    id integer NOT NULL
);


ALTER TABLE model OWNER TO postgres;

--
-- Name: model_id_seq; Type: SEQUENCE; Schema: public; Owner: postgres
--

CREATE SEQUENCE model_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


ALTER TABLE model_id_seq OWNER TO postgres;

--
-- Name: model_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE model_id_seq OWNED BY model.id;


--
-- Name: residue; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE residue (
    model character varying(30),
    residue character varying(3),
    reslabel integer,
    resorder integer,
    id integer NOT NULL
);

--
-- Name: redundancy; Type: TABLE; Schema: public; Owner: postgres
--

CREATE TABLE redundancy (
    model character(30) NOT NULL,
    match character(30) NOT NULL
);

ALTER TABLE redundancy OWNER TO postgres;

ALTER TABLE residue_id_seq OWNER TO postgres;

--
-- Name: residue_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: postgres
--

ALTER SEQUENCE residue_id_seq OWNED BY residue.id;


--
-- Name: angle id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY angle ALTER COLUMN id SET DEFAULT nextval('angle_id_seq'::regclass);


--
-- Name: atom id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY atom ALTER COLUMN id SET DEFAULT nextval('atom_id_seq'::regclass);


--
-- Name: model id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY model ALTER COLUMN id SET DEFAULT nextval('model_id_seq'::regclass);


--
-- Name: residue id; Type: DEFAULT; Schema: public; Owner: postgres
--

ALTER TABLE ONLY residue ALTER COLUMN id SET DEFAULT nextval('residue_id_seq'::regclass);

create unique index on model(code);
create index on residue(model); 
create index on atom(model); 
create index on redundancy(model); 
create index on angle(model); 

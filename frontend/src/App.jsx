import { useEffect, useMemo, useRef, useState } from "react";
import {
  AppBar, Toolbar, IconButton, Box, Container, Paper, Typography, Stack,
  TextField, Button, Chip, Select, MenuItem, LinearProgress, Avatar,
  Grid, Divider, Tooltip, LinearProgress as Progress, Card, CardContent
} from "@mui/material";
import { useTheme } from "@mui/material/styles";
import BusinessCenterRoundedIcon from "@mui/icons-material/BusinessCenterRounded";
import SmartToyRoundedIcon from "@mui/icons-material/SmartToyRounded";
import PersonRoundedIcon from "@mui/icons-material/PersonRounded";
import SendRoundedIcon from "@mui/icons-material/Send";
import ContentCopyRoundedIcon from "@mui/icons-material/ContentCopyRounded";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

const API = import.meta.env.VITE_API_URL || "http://localhost:8000";

const QUICK = [
  "What is NEC?",
  "Which article defines grounded conductor?",
  "Explain equipment grounding conductor.",
  "What services does Wattmonk offer?",
];

export default function App({ preset = "indigo", setPreset = () => {} }) {
  const theme = useTheme();
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [domain, setDomain] = useState("auto");
  const [loading, setLoading] = useState(false);
  const [stats, setStats] = useState(null);

  const sessionRef = useRef(localStorage.getItem("sid") || "");
  const scrollRef = useRef(null);

  useEffect(() => {
    if (!sessionRef.current) {
      sessionRef.current = crypto.randomUUID();
      localStorage.setItem("sid", sessionRef.current);
    }
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  useEffect(() => {
    fetch(`${API}/stats`).then(r => r.json()).then(setStats).catch(() => {});
  }, []);

  const uniqueSources = (sources = []) => {
    const seen = new Set(); const out = [];
    for (const s of sources) {
      const key = s.title || s.doc_id;
      if (!seen.has(key)) { seen.add(key); out.push(s); }
    }
    return out;
  };

  const send = async (overrideText) => {
    const text = (overrideText ?? input).trim();
    if (!text || loading) return;
    setLoading(true);
    setMessages((m) => [...m, { role: "user", content: text }]);
    setInput("");
    try {
      const r = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, session_id: sessionRef.current, domain }),
      });
      if (!r.ok) throw new Error(await r.text());
      const data = await r.json();
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: data?.answer ?? "(No answer)",
          sources: Array.isArray(data?.sources) ? data.sources : [],
          confidence: typeof data?.confidence === "number" ? data.confidence : undefined,
        },
      ]);
    } catch (e) {
      setMessages((m) => [...m, { role: "assistant", content: `Error: ${e.message || e}` }]);
    } finally { setLoading(false); }
  };

  const onKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  const Header = (
    <AppBar position="sticky" color="transparent" elevation={0} sx={{ py: 1 }}>
      <Toolbar>
        <BusinessCenterRoundedIcon sx={{ mr: 1.2 }} />
        <Typography variant="h6" fontWeight={800}>Business RAG Chatbot</Typography>
        <Box sx={{ flex: 1 }} />
        <Typography variant="body2" color="text.secondary" sx={{ mr: 2 }}>
          NEC • Wattmonk • General
        </Typography>
        {/* Palette switcher */}
        <Select
          size="small"
          value={preset}
          onChange={(e) => setPreset(e.target.value)}
          sx={{ minWidth: 140, bgcolor: "background.paper" }}
        >
          <MenuItem value="indigo">Indigo</MenuItem>
          <MenuItem value="sapphire">Sapphire</MenuItem>
          <MenuItem value="charcoal">Charcoal</MenuItem>
          <MenuItem value="warm">Warm</MenuItem>
          <MenuItem value="dark">Dark</MenuItem>
        </Select>
      </Toolbar>
    </AppBar>
  );

  const EmptyState = (
    <Paper sx={{ p: 4, textAlign: "center" }}>
      <Avatar sx={{ width: 56, height: 56, bgcolor: "primary.main", mx: "auto", mb: 2 }}>
        <SmartToyRoundedIcon />
      </Avatar>
      <Typography variant="h6" fontWeight={800}>Ask about NEC or Wattmonk</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ maxWidth: 560, mx: "auto", mt: 1 }}>
        This assistant uses Retrieval-Augmented Generation (RAG). Add PDFs/DOCX to <code>backend/data/*</code>,
        run <b>/ingest</b>, then ask domain questions to get answers with citations.
      </Typography>
      <Stack direction="row" spacing={1} sx={{ mt: 2 }} justifyContent="center" flexWrap="wrap">
        {QUICK.map((q) => <Chip key={q} label={q} onClick={() => send(q)} />)}
      </Stack>
      <Divider sx={{ my: 2 }} />
      <Typography variant="caption" color="text.secondary">
        Tip: Use the selector (Auto / NEC / Wattmonk / General) to guide intent.
      </Typography>
    </Paper>
  );

  const SourceCard = ({ s }) => (
    <Card variant="outlined" sx={{ minWidth: 220, maxWidth: 320 }}>
      <CardContent sx={{ py: 1.25 }}>
        <Typography variant="caption" noWrap title={s.title}>{s.title}</Typography>
        <Progress variant="determinate" value={Math.round((s.score ?? 0) * 100)} sx={{ mt: 1, height: 6, borderRadius: 5 }} />
      </CardContent>
    </Card>
  );

  const Bubble = ({ m }) => (
    <Stack direction={m.role === "user" ? "row-reverse" : "row"} spacing={1.25} alignItems="flex-start">
      <Avatar
        sx={{
          bgcolor: m.role === "user" ? "primary.main" : "grey.200",
          color: m.role === "user" ? "primary.contrastText" : "text.primary",
          width: 34, height: 34,
        }}
      >
        {m.role === "user" ? <PersonRoundedIcon /> : <SmartToyRoundedIcon fontSize="small" />}
      </Avatar>
      <Box sx={{
        display: "inline-block",
        p: 1.75, px: 2,
        maxWidth: "80%",
        borderRadius: m.role === "user" ? "18px 18px 4px 18px" : "18px 18px 18px 4px",
        boxShadow: 1,
        ...(m.role === "user"
          ? { bgcolor: "primary.main", color: "primary.contrastText" }
          : { bgcolor: "background.paper" }),
      }}>
        <Typography component="div" variant="body2" sx={{ whiteSpace: "pre-wrap" }}>
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{m.content}</ReactMarkdown>
        </Typography>

        {m.role === "assistant" && (
          <Stack direction="row" spacing={1} alignItems="center" sx={{ mt: 1 }}>
            {typeof m.confidence === "number" && (
              <Chip size="small" label={`Confidence: ${(m.confidence * 100).toFixed(0)}%`} />
            )}
            <Tooltip title="Copy answer">
              <IconButton size="small" onClick={() => navigator.clipboard.writeText(m.content)}>
                <ContentCopyRoundedIcon fontSize="inherit" />
              </IconButton>
            </Tooltip>
          </Stack>
        )}

        {Array.isArray(m.sources) && m.sources.length > 0 && (
          <Stack direction="row" spacing={1} mt={1.25} flexWrap="wrap">
            {uniqueSources(m.sources).slice(0, 3).map((s) => <SourceCard key={s.doc_id} s={s} />)}
          </Stack>
        )}
      </Box>
    </Stack>
  );

  return (
    <Box sx={{ minHeight: "100vh", background: theme.custom?.appBg || theme.palette.background.default }}>
      {Header}

      <Container maxWidth="lg" sx={{ py: 3 }}>
        <Grid
          container
          columnSpacing={3}
          rowSpacing={{ xs: 2, md: 0 }}
          alignItems="flex-start"
        >
          {/* Sidebar */}
          <Grid
            item xs={12} md={3}
            sx={{
              position: { md: "sticky" },
              top: { md: 84 },
              alignSelf: "flex-start",
              zIndex: 1,
            }}
          >
            <Paper sx={{ p: 2.25 }}>
              <Typography variant="overline" color="text.secondary">WORKSPACE</Typography>
              <Typography variant="h6" fontWeight={800} sx={{ mt: 0.5 }}>NEC / Wattmonk KB</Typography>

              <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" sx={{ mt: 1, mb: 1 }}>
                <Chip label="Auto"       color={domain==="auto"?"primary":undefined}      onClick={()=>setDomain("auto")}       size="small" />
                <Chip label="NEC"        color={domain==="nec"?"primary":undefined}       onClick={()=>setDomain("nec")}        size="small" />
                <Chip label="Wattmonk"   color={domain==="wattmonk"?"primary":undefined}  onClick={()=>setDomain("wattmonk")}   size="small" />
                <Chip label="General"    color={domain==="general"?"primary":undefined}   onClick={()=>setDomain("general")}    size="small" />
              </Stack>

              <Divider sx={{ my: 1.5 }} />

              <Typography variant="overline" color="text.secondary">STATUS</Typography>
              <Stack spacing={0.75} sx={{ mt: 0.75 }}>
                <Row label="Docs (total)"     value={stats?.total ?? "—"} />
                <Row label="NEC chunks"       value={stats?.nec_count ?? "—"} />
                <Row label="Wattmonk chunks"  value={stats?.wattmonk_count ?? "—"} />
              </Stack>

              <Divider sx={{ my: 1.5 }} />

              <Typography variant="overline" color="text.secondary">QUICK ASKS</Typography>
              <Stack direction="row" useFlexGap flexWrap="wrap" spacing={1} sx={{ mt: 1 }}>
                {QUICK.map((q) => (
                  <Chip key={q} label={q} onClick={() => send(q)} size="small" sx={{ borderRadius: 999 }} />
                ))}
              </Stack>
            </Paper>
          </Grid>

          {/* Chat area */}
          <Grid item xs={12} md={9}>
            <Paper ref={scrollRef} sx={{ p: 2.5, height: { xs: "60vh", md: "68vh" }, overflow: "auto" }}>
              {messages.length === 0 && !loading ? (
                EmptyState
              ) : (
                <Stack spacing={2}>
                  {messages.map((m, i) => <Bubble key={i} m={m} />)}
                  {loading && <LinearProgress />}
                </Stack>
              )}
            </Paper>

            {/* Composer */}
            <Paper elevation={0} sx={{ mt: 2, p: 1, display: "flex", gap: 1, alignItems: "center" }}>
              <Select size="small" value={domain} onChange={(e)=>setDomain(e.target.value)} disabled={loading} sx={{ minWidth: 140 }}>
                <MenuItem value="auto">Auto</MenuItem>
                <MenuItem value="nec">NEC</MenuItem>
                <MenuItem value="wattmonk">Wattmonk</MenuItem>
                <MenuItem value="general">General</MenuItem>
              </Select>
              <TextField
                fullWidth size="small" placeholder="Ask a question…"
                value={input} onChange={(e)=>setInput(e.target.value)}
                onKeyDown={onKeyDown} disabled={loading}
              />
              <Button variant="contained" onClick={()=>send()} disabled={loading} endIcon={<SendRoundedIcon />}>
                {loading ? "Sending…" : "Send"}
              </Button>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </Box>
  );
}

// tiny row component
function Row({ label, value }) {
  const muted = useMemo(() => ({ color: "text.secondary" }), []);
  return (
    <Stack direction="row" justifyContent="space-between">
      <Typography variant="body2" sx={muted}>{label}</Typography>
      <Typography variant="body2" fontWeight={600}>{value}</Typography>
    </Stack>
  );
}

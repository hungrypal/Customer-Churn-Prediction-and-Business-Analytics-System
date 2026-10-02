import { useState } from "react";
import { Link, Navigate, useLocation, useNavigate } from "react-router-dom";
import { motion } from "motion/react";
import { ArrowRight, ShieldAlert } from "lucide-react";
import { useAuth } from "./AuthContext";

function authError(error) {
  if (error?.status === 409) return "An account with this email already exists.";
  if (error?.status === 401) return "Invalid email or password.";
  if (error?.status === 0 || error?.status >= 500) return "The authentication service is unavailable.";
  return error?.message || "Please check your details and try again.";
}

function AuthShell({ eyebrow, title, subtitle, children, footer }) {
  return <main className="auth-page"><motion.section className="auth-card" initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3 }}><Link className="auth-brand" to="/"><span><ShieldAlert size={19} /></span>Churn<span>Signal</span></Link><p className="eyebrow">{eyebrow}</p><h1>{title}</h1><p className="auth-subtitle">{subtitle}</p>{children}<div className="auth-footer">{footer}</div></motion.section></main>;
}

export function LoginPage() {
  const auth = useAuth(); const navigate = useNavigate(); const location = useLocation(); const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const [error, setError] = useState(""); const [loading, setLoading] = useState(false);
  if (auth.status === "loading") return <div className="auth-state">Checking your session...</div>;
  if (auth.authenticated) return <Navigate to={location.state?.from || "/"} replace />;
  async function submit(event) { event.preventDefault(); setError(""); setLoading(true); try { await auth.login(email.trim(), password); navigate(location.state?.from || "/", { replace: true }); } catch (requestError) { setError(authError(requestError)); } finally { setLoading(false); } }
  return <AuthShell eyebrow="WELCOME BACK" title="Sign in to ChurnSignal" subtitle="Use your account to access prediction intelligence and retention analytics." footer={<>New here? <Link to="/signup">Create an account</Link></>}><form className="auth-form" onSubmit={submit} noValidate><label>Email<input type="email" autoComplete="email" required value={email} onChange={(event) => setEmail(event.target.value)} /></label><label>Password<input type="password" autoComplete="current-password" required value={password} onChange={(event) => setPassword(event.target.value)} /></label>{(error || auth.status === "unavailable") && <motion.p className="auth-error" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>{error || "The authentication service is unavailable."}</motion.p>}<button className="button auth-submit" disabled={loading}>{loading ? "Signing in..." : <>Sign in <ArrowRight size={16} /></>}</button></form><p className="auth-note">Your session uses a secure HTTP-only cookie.</p></AuthShell>;
}

export function SignupPage() {
  const auth = useAuth(); const navigate = useNavigate(); const [name, setName] = useState(""); const [email, setEmail] = useState(""); const [password, setPassword] = useState(""); const [confirm, setConfirm] = useState(""); const [error, setError] = useState(""); const [loading, setLoading] = useState(false);
  if (auth.status === "loading") return <div className="auth-state">Checking your session...</div>;
  if (auth.authenticated) return <Navigate to="/" replace />;
  async function submit(event) { event.preventDefault(); setError(""); if (password.length < 8) { setError("Password must be at least 8 characters."); return; } if (password !== confirm) { setError("Passwords do not match."); return; } setLoading(true); try { await auth.signup(name.trim(), email.trim(), password); navigate("/", { replace: true }); } catch (requestError) { setError(authError(requestError)); } finally { setLoading(false); } }
  return <AuthShell eyebrow="GET STARTED" title="Create your account" subtitle="Build a focused workflow for understanding and reducing customer churn." footer={<>Already have an account? <Link to="/login">Sign in</Link></>}><form className="auth-form" onSubmit={submit} noValidate><label>Name<input type="text" autoComplete="name" required value={name} onChange={(event) => setName(event.target.value)} /></label><label>Email<input type="email" autoComplete="email" required value={email} onChange={(event) => setEmail(event.target.value)} /></label><label>Password<input type="password" autoComplete="new-password" minLength={8} required value={password} onChange={(event) => setPassword(event.target.value)} /></label><label>Confirm password<input type="password" autoComplete="new-password" minLength={8} required value={confirm} onChange={(event) => setConfirm(event.target.value)} /></label>{(error || auth.status === "unavailable") && <motion.p className="auth-error" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>{error || "The authentication service is unavailable."}</motion.p>}<button className="button auth-submit" disabled={loading}>{loading ? "Creating account..." : <>Create account <ArrowRight size={16} /></>}</button></form></AuthShell>;
}

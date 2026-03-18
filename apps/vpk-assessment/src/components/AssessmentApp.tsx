"use client";

import Image from "next/image";
import { FormEvent, useEffect, useMemo, useState } from "react";

import { PieResultsChart } from "@/components/PieResultsChart";
import type {
  AttemptSnapshot,
  AuthenticatedAccount,
  ManagedAccountSummary,
  QuestionPayload,
  ResultPayload,
  WizardStage,
} from "@/lib/types";
import { countryPhoneOptions, deriveAgeFromDateOfBirth } from "@/lib/validation";

type AssessmentAppProps = {
  initialAccount: AuthenticatedAccount | null;
  initialAccounts: ManagedAccountSummary[];
  initialSnapshot: AttemptSnapshot | null;
  initialQuestion: QuestionPayload | null;
  initialResult: ResultPayload | null;
};

type IdentityForm = {
  firstName: string;
  middleName: string;
  lastName: string;
  dateOfBirth: string;
  location: string;
  email: string;
  countryCode: string;
  localPhoneNumber: string;
};

type LoginForm = {
  username: string;
  password: string;
};

type AccountCreationForm = {
  displayName: string;
  username: string;
  password: string;
};

const emptyForm: IdentityForm = {
  firstName: "",
  middleName: "",
  lastName: "",
  dateOfBirth: "",
  location: "",
  email: "",
  countryCode: "IN",
  localPhoneNumber: "",
};

const emptyLoginForm: LoginForm = {
  username: "",
  password: "",
};

const emptyAccountForm: AccountCreationForm = {
  displayName: "",
  username: "",
  password: "",
};

const openingHeadline = "AYURVEDIC PRAKRITI ASSESSMENT";
const openingSummaryLines = [
  "Secure login, guided instructions, one category at a time, and a final",
  "constitution result view only after completion.",
];
const verseSanskrit = [
  "नमामि धन्वन्तरिमादिदेवं सुरासुरैर्वन्दितपादपद्मम् ।",
  "लोके जरारुग्भयमृत्युनाशनं धातारमीशं विविधौषधीनाम् ॥",
];
const verseTransliteration = [
  "Namami Dhanvantari Adidevam Surasurairvanditapadapadmam I",
  "Loke Jararugbhayamrityunashanam Dhataramisham Vividhauṣadhinam II",
];

function isEditableTarget(target: EventTarget | null) {
  return target instanceof HTMLElement
    ? Boolean(target.closest("input, textarea, select, [contenteditable='true']"))
    : false;
}

function describeWindowStatus(account: ManagedAccountSummary) {
  if (account.role === "admin" || account.windowStatus === "unlimited") {
    return "Unlimited access";
  }

  if (account.windowStatus === "used") {
    return "Test already completed";
  }

  if (account.windowStatus === "not-started") {
    return "Window will begin at the next eligible login";
  }

  if (account.windowStatus === "expired") {
    return account.accessWindowExpiresAt
      ? `Window expired on ${new Date(account.accessWindowExpiresAt).toLocaleString()}`
      : "Window expired";
  }

  return account.accessWindowExpiresAt
    ? `Window open until ${new Date(account.accessWindowExpiresAt).toLocaleString()}`
    : "Window active";
}

export function AssessmentApp({
  initialAccount,
  initialAccounts,
  initialSnapshot,
  initialQuestion,
  initialResult,
}: AssessmentAppProps) {
  const [account] = useState<AuthenticatedAccount | null>(initialAccount);
  const [stage, setStage] = useState<WizardStage>(initialSnapshot?.stage ?? "opening");
  const [resumeStage, setResumeStage] = useState<WizardStage | null>(
    initialSnapshot?.stage && initialSnapshot.stage !== "opening" ? initialSnapshot.stage : null,
  );
  const [attemptId, setAttemptId] = useState<string | null>(
    initialSnapshot?.attemptId ?? initialResult?.attemptId ?? null,
  );
  const [questionIndex, setQuestionIndex] = useState(initialSnapshot?.questionIndex ?? 1);
  const [identityForm, setIdentityForm] = useState(emptyForm);
  const [instructionsText, setInstructionsText] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [uiError, setUiError] = useState("");
  const [duplicateMessage, setDuplicateMessage] = useState("");
  const [question, setQuestion] = useState<QuestionPayload | null>(initialQuestion);
  const [questionLoading, setQuestionLoading] = useState(false);
  const [lifetimeSelection, setLifetimeSelection] = useState<string | null>(
    initialQuestion?.savedResponse.lifetimeOptionId ?? null,
  );
  const [presentSelection, setPresentSelection] = useState<string | null>(
    initialQuestion?.savedResponse.presentOptionId ?? null,
  );
  const [result, setResult] = useState<ResultPayload | null>(initialResult);
  const [loginForm, setLoginForm] = useState<LoginForm>(emptyLoginForm);
  const [loginErrors, setLoginErrors] = useState<Record<string, string>>({});
  const [loginError, setLoginError] = useState("");
  const [loginSubmitting, setLoginSubmitting] = useState(false);
  const [adminPanelOpen, setAdminPanelOpen] = useState(false);
  const [accountForm, setAccountForm] = useState<AccountCreationForm>(emptyAccountForm);
  const [accountFormErrors, setAccountFormErrors] = useState<Record<string, string>>({});
  const [accountFormError, setAccountFormError] = useState("");
  const [accountFormSuccess, setAccountFormSuccess] = useState("");
  const [accountSubmitting, setAccountSubmitting] = useState(false);
  const [accountActionError, setAccountActionError] = useState("");
  const [accountActionSuccess, setAccountActionSuccess] = useState("");
  const [retestSubmittingId, setRetestSubmittingId] = useState<string | null>(null);
  const [managedAccounts, setManagedAccounts] = useState<ManagedAccountSummary[]>(initialAccounts);
  const registrantName = initialSnapshot?.registrantName?.trim() || account?.displayName || "the registered person";
  const accessWindowExpiresAt = initialSnapshot?.accessWindowExpiresAt ?? null;
  const protectionsActive = account?.role !== "admin";
  const derivedAge = identityForm.dateOfBirth
    ? deriveAgeFromDateOfBirth(identityForm.dateOfBirth)
    : null;
  const watermarkSeed = useMemo(
    () =>
      account && protectionsActive
        ? Array.from({ length: 12 }, (_, index) => `${account.username} • ${account.displayName} • Private VPK • ${index + 1}`)
        : [],
    [account, protectionsActive],
  );

  useEffect(() => {
    if (!account || !protectionsActive) {
      return;
    }

    const preventProtectedAction = (event: Event) => {
      if (!isEditableTarget(event.target)) {
        event.preventDefault();
      }
    };

    const handleKeydown = (event: KeyboardEvent) => {
      const lowerKey = event.key.toLowerCase();
      const modifierPressed = event.metaKey || event.ctrlKey;

      if (modifierPressed && ["c", "x", "a", "s", "p"].includes(lowerKey) && !isEditableTarget(event.target)) {
        event.preventDefault();
      }

      if (event.metaKey && event.shiftKey && ["3", "4", "5"].includes(lowerKey)) {
        event.preventDefault();
      }
    };

    document.addEventListener("copy", preventProtectedAction);
    document.addEventListener("cut", preventProtectedAction);
    document.addEventListener("contextmenu", preventProtectedAction);
    document.addEventListener("dragstart", preventProtectedAction);
    document.addEventListener("selectstart", preventProtectedAction);
    document.addEventListener("keydown", handleKeydown);

    return () => {
      document.removeEventListener("copy", preventProtectedAction);
      document.removeEventListener("cut", preventProtectedAction);
      document.removeEventListener("contextmenu", preventProtectedAction);
      document.removeEventListener("dragstart", preventProtectedAction);
      document.removeEventListener("selectstart", preventProtectedAction);
      document.removeEventListener("keydown", handleKeydown);
    };
  }, [account, protectionsActive]);

  function updateField<Key extends keyof IdentityForm>(key: Key, value: IdentityForm[Key]) {
    setIdentityForm((current) => ({ ...current, [key]: value }));
  }

  function updateLoginField<Key extends keyof LoginForm>(key: Key, value: LoginForm[Key]) {
    setLoginForm((current) => ({ ...current, [key]: value }));
  }

  function updateAccountField<Key extends keyof AccountCreationForm>(
    key: Key,
    value: AccountCreationForm[Key],
  ) {
    setAccountForm((current) => ({ ...current, [key]: value }));
  }

  function goToStage(nextStage: WizardStage) {
    if (nextStage !== "opening") {
      setResumeStage(nextStage);
    }
    setAdminPanelOpen(false);
    setStage(nextStage);
  }

  function handleHome() {
    if (stage !== "opening") {
      setResumeStage(stage);
    }
    setAdminPanelOpen(false);
    setUiError("");
    setStage("opening");
  }

  async function handleLoginSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoginSubmitting(true);
    setLoginErrors({});
    setLoginError("");

    try {
      const response = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(loginForm),
      });
      const payload = await response.json();

      if (!response.ok) {
        if (payload.fieldErrors) {
          setLoginErrors(payload.fieldErrors);
          return;
        }

        setLoginError(payload.error ?? "Unable to log in.");
        return;
      }

      window.location.reload();
    } finally {
      setLoginSubmitting(false);
    }
  }

  async function handleLogout() {
    await fetch("/api/auth/logout", { method: "POST" });
    window.location.reload();
  }

  async function enterAssessment(index = questionIndex) {
    if (question) {
      goToStage("assessment");
      return;
    }

    const loaded = await loadQuestion(index);
    if (loaded) {
      goToStage("assessment");
    }
  }

  async function handleOpeningContinue() {
    if (resumeStage && resumeStage !== "opening") {
      if (resumeStage === "assessment" && !question) {
        await enterAssessment(questionIndex);
        return;
      }
      goToStage(resumeStage);
      return;
    }

    goToStage("identity");
  }

  async function loadQuestion(index: number) {
    setQuestionLoading(true);
    setUiError("");

    try {
      const response = await fetch(`/api/questions/${index}`);
      const payload = await response.json();

      if (!response.ok) {
        setUiError(payload.error ?? "Unable to load this question.");
        return false;
      }

      setQuestion(payload);
      setQuestionIndex(payload.index);
      setLifetimeSelection(payload.savedResponse.lifetimeOptionId);
      setPresentSelection(payload.savedResponse.presentOptionId);
      return true;
    } catch {
      setUiError("Unable to load this question.");
      return false;
    } finally {
      setQuestionLoading(false);
    }
  }

  useEffect(() => {
    if (!account) {
      return;
    }

    if (stage === "instructions" && !instructionsText) {
      fetch("/api/instructions")
        .then((response) => response.json())
        .then((payload: { instructionText: string }) => {
          setInstructionsText(payload.instructionText);
        })
        .catch(() => {
          setUiError("Unable to load instructions.");
        });
    }
  }, [account, instructionsText, stage]);

  useEffect(() => {
    if (!account) {
      return;
    }

    if (stage === "assessment" && !question && !questionLoading) {
      void loadQuestion(questionIndex);
    }
  }, [account, question, questionIndex, questionLoading, stage]);

  async function handleIdentitySubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setUiError("");
    setErrors({});

    const response = await fetch("/api/identity", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(identityForm),
    });
    const payload = await response.json();

    if (!response.ok) {
      if (payload.fieldErrors) {
        setErrors(payload.fieldErrors);
        return;
      }

      if (payload.duplicate) {
        setDuplicateMessage(payload.message);
        goToStage("duplicate");
        return;
      }

      setUiError(payload.error ?? "Unable to save your details.");
      return;
    }

    setAttemptId(payload.attemptId);
    goToStage("instructions");
  }

  async function handleAcknowledge() {
    const response = await fetch("/api/instructions/acknowledge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ attemptId, acknowledged: true }),
    });
    const payload = await response.json();

    if (!response.ok) {
      setUiError(payload.error ?? "Unable to record your acknowledgement.");
      return;
    }

    goToStage("start");
  }

  async function handleNext() {
    if (!question || !lifetimeSelection || !presentSelection) {
      setUiError("Select one option for Lifetime and one option for Present before continuing.");
      return;
    }

    const response = await fetch(`/api/questions/${question.category.id}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        attemptId,
        lifetimeOptionId: lifetimeSelection,
        presentOptionId: presentSelection,
      }),
    });
    const payload = await response.json();

    if (!response.ok) {
      setUiError(payload.error ?? "Unable to save this response.");
      return;
    }

    if (question.index === question.total) {
      await handleComplete();
      return;
    }

    setQuestion(null);
    await loadQuestion(payload.nextIndex);
  }

  async function handleBack() {
    const nextIndex = Math.max(1, questionIndex - 1);
    setQuestion(null);
    await loadQuestion(nextIndex);
  }

  async function handleComplete() {
    const response = await fetch("/api/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ attemptId }),
    });
    const payload = await response.json();

    if (!response.ok) {
      setUiError(payload.error ?? "Unable to complete the assessment.");
      return;
    }

    setResult(payload);
    goToStage("result");
  }

  async function handleCreateAccount(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setAccountSubmitting(true);
    setAccountFormErrors({});
    setAccountFormError("");
    setAccountFormSuccess("");
    setAccountActionError("");
    setAccountActionSuccess("");

    try {
      const response = await fetch("/api/admin/accounts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(accountForm),
      });
      const payload = await response.json();

      if (!response.ok) {
        if (payload.fieldErrors) {
          setAccountFormErrors(payload.fieldErrors);
          return;
        }

        setAccountFormError(payload.error ?? "Unable to create the account.");
        return;
      }

      setManagedAccounts((current) => [...current, payload.account]);
      setAccountForm(emptyAccountForm);
      setAccountFormSuccess(`Account ${payload.account.username} is ready to log in.`);
    } finally {
      setAccountSubmitting(false);
    }
  }

  async function handleAllowRetest(accountId: string) {
    setRetestSubmittingId(accountId);
    setAccountActionError("");
    setAccountActionSuccess("");

    try {
      const response = await fetch("/api/admin/accounts", {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ accountId }),
      });
      const payload = await response.json();

      if (!response.ok) {
        setAccountActionError(payload.error ?? "Unable to update the account.");
        return;
      }

      setManagedAccounts(payload.accounts ?? managedAccounts);
      const refreshedAccount = (payload.accounts ?? managedAccounts).find(
        (managedAccount: ManagedAccountSummary) => managedAccount.id === accountId,
      );
      setAccountActionSuccess(
        refreshedAccount
          ? `${refreshedAccount.displayName} can log in again for a fresh 6-hour test window.`
          : "The account has been updated.",
      );
    } finally {
      setRetestSubmittingId(null);
    }
  }

  if (!account) {
    return (
      <main className="app-shell app-shell--opening">
        <div className="app-shell__inner">
          <section className="opening-stage opening-stage--login">
            <div className="opening-stage__backdrop" aria-hidden="true">
              <Image
                src="/dhanvantri-opening.png"
                alt=""
                fill
                priority
                className="opening-stage__image"
                sizes="100vw"
              />
            </div>
            <div className="opening-stage__layout opening-stage__layout--login">
              <div className="opening-stage__zone opening-stage__zone--left">
                <div className="opening-stage__copy">
                  <p className="opening-stage__lead">Traditional constitutional assessment</p>
                  <h1 className="opening-stage__title" aria-label={openingHeadline}>
                    <span>AYURVEDIC</span>
                    <span>PRAKRITI</span>
                    <span>ASSESSMENT</span>
                  </h1>
                  <div className="opening-stage__summary">
                    {openingSummaryLines.map((line) => (
                      <p key={line}>{line}</p>
                    ))}
                  </div>
                  <div className="opening-stage__verse">
                    <div className="opening-stage__sanskrit" lang="sa">
                      {verseSanskrit.map((line) => (
                        <p key={line}>{line}</p>
                      ))}
                    </div>
                    <div className="opening-stage__transliteration">
                      {verseTransliteration.map((line) => (
                        <p key={line}>{line}</p>
                      ))}
                    </div>
                    <p className="opening-stage__note">Enter with the credentials assigned for this assessment.</p>
                  </div>
                </div>
              </div>
              <div className="opening-stage__zone opening-stage__zone--right opening-stage__zone--login-panel">
                <div className="panel stack opening-stage__login-panel">
                  <form className="form-grid" onSubmit={handleLoginSubmit}>
                    <div className="field">
                      <label className="sr-only" htmlFor="login-username">Username</label>
                      <input
                        className="input"
                        id="login-username"
                        autoComplete="username"
                        placeholder="Username"
                        value={loginForm.username}
                        onChange={(event) => updateLoginField("username", event.target.value)}
                      />
                      {loginErrors.username ? <p className="error-text">{loginErrors.username}</p> : null}
                    </div>
                    <div className="field">
                      <label className="sr-only" htmlFor="login-password">Password</label>
                      <input
                        className="input"
                        id="login-password"
                        type="password"
                        autoComplete="current-password"
                        placeholder="Password"
                        value={loginForm.password}
                        onChange={(event) => updateLoginField("password", event.target.value)}
                      />
                      {loginErrors.password ? <p className="error-text">{loginErrors.password}</p> : null}
                    </div>
                    {loginError ? <p className="error-text">{loginError}</p> : null}
                    <div className="button-row">
                      <button className="button button--primary" type="submit" disabled={loginSubmitting}>
                        {loginSubmitting ? "Signing in..." : "Login"}
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>
    );
  }

  return (
    <main
      className={`app-shell ${stage === "opening" ? "app-shell--opening" : ""} ${protectionsActive ? "app-shell--protected" : ""}`}
    >
      {protectionsActive ? (
        <div className="privacy-watermark" aria-hidden="true">
          {watermarkSeed.map((label) => (
            <span key={label}>{label}</span>
          ))}
        </div>
      ) : null}
      <div className={`app-shell__toolbar ${stage === "opening" ? "app-shell__toolbar--opening" : ""}`}>
        <button
          className={`button button--home ${stage === "opening" && !adminPanelOpen ? "button--home-active" : ""}`}
          type="button"
          onClick={handleHome}
          aria-current={stage === "opening" && !adminPanelOpen ? "page" : undefined}
        >
          Home
        </button>
        {account.role === "admin" ? (
          <button
            className={`button button--secondary ${adminPanelOpen ? "button--home-active" : ""}`}
            type="button"
            onClick={() => setAdminPanelOpen((current) => !current)}
          >
            {adminPanelOpen ? "Close Accounts" : "Manage Accounts"}
          </button>
        ) : null}
        <button className="button button--ghost" type="button" onClick={handleLogout}>
          Logout
        </button>
      </div>
      <div className="app-shell__inner">
        {adminPanelOpen ? (
          <>
            <section className="hero hero--compact">
              <span className="eyebrow">Admin Controls</span>
              <h1 className="hero__single-line">Create assessment accounts and hand out passwords carefully.</h1>
              <p>
                This panel is visible only to the admin login. Accounts created here can sign in, complete the assessment, and the admin can reopen access when another attempt is approved.
              </p>
            </section>

            <section className="split split--admin">
              <div className="panel stack">
                <div className="stack">
                  <h2 className="section-title">Add account</h2>
                  <p className="muted">Create a login name and password for the person who will take the test.</p>
                </div>
                <form className="form-grid" onSubmit={handleCreateAccount}>
                  <div className="field">
                    <label htmlFor="account-display-name">Account holder name</label>
                    <input
                      className="input"
                      id="account-display-name"
                      value={accountForm.displayName}
                      onChange={(event) => updateAccountField("displayName", event.target.value)}
                    />
                    {accountFormErrors.displayName ? <p className="error-text">{accountFormErrors.displayName}</p> : null}
                  </div>
                  <div className="field-grid field-grid--double">
                    <div className="field">
                      <label htmlFor="account-username">Username</label>
                      <input
                        className="input"
                        id="account-username"
                        value={accountForm.username}
                        onChange={(event) => updateAccountField("username", event.target.value)}
                      />
                      {accountFormErrors.username ? <p className="error-text">{accountFormErrors.username}</p> : null}
                    </div>
                    <div className="field">
                      <label htmlFor="account-password">Password</label>
                      <input
                        className="input"
                        id="account-password"
                        type="password"
                        value={accountForm.password}
                        onChange={(event) => updateAccountField("password", event.target.value)}
                      />
                      {accountFormErrors.password ? <p className="error-text">{accountFormErrors.password}</p> : null}
                    </div>
                  </div>
                  {accountFormError ? <p className="error-text">{accountFormError}</p> : null}
                  {accountFormSuccess ? <p className="success-text">{accountFormSuccess}</p> : null}
                  <div className="button-row">
                    <button className="button button--primary" type="submit" disabled={accountSubmitting}>
                      {accountSubmitting ? "Creating..." : "Create Account"}
                    </button>
                  </div>
                </form>
              </div>

              <div className="panel stack">
                <div className="stack">
                  <h2 className="section-title">Existing accounts</h2>
                  <p className="muted">Share usernames and passwords only with the intended assessment holder.</p>
                </div>
                {accountActionError ? <p className="error-text">{accountActionError}</p> : null}
                {accountActionSuccess ? <p className="success-text">{accountActionSuccess}</p> : null}
                <div className="account-list">
                  {managedAccounts.map((managedAccount) => (
                    <article className="account-card" key={managedAccount.id}>
                      <div className="account-card__header">
                        <strong>{managedAccount.displayName}</strong>
                        <span className="constitution-pill">{managedAccount.role}</span>
                      </div>
                      <p className="muted">Username: {managedAccount.username}</p>
                      <p className="muted">Created: {new Date(managedAccount.createdAt).toLocaleString()}</p>
                      <p className="muted">
                        Last login: {managedAccount.lastLoginAt ? new Date(managedAccount.lastLoginAt).toLocaleString() : "Not yet used"}
                      </p>
                      {managedAccount.role === "user" ? (
                        <>
                          <p className="muted">Attempts used: {managedAccount.attemptsUsed}</p>
                          <p className="muted">Completed tests: {managedAccount.completedAttempts}</p>
                          <p className="muted">Available attempts: {managedAccount.availableAttempts}</p>
                          <p className="muted">{describeWindowStatus(managedAccount)}</p>
                          <div className="button-row">
                            <button
                              className="button button--secondary"
                              type="button"
                              onClick={() => void handleAllowRetest(managedAccount.id)}
                              disabled={retestSubmittingId === managedAccount.id}
                            >
                              {retestSubmittingId === managedAccount.id ? "Updating..." : "Allow Test Again"}
                            </button>
                          </div>
                        </>
                      ) : null}
                    </article>
                  ))}
                </div>
              </div>
            </section>
          </>
        ) : stage === "opening" ? (
          <section className="opening-stage">
            <div className="opening-stage__backdrop" aria-hidden="true">
              <Image
                src="/dhanvantri-opening.png"
                alt=""
                fill
                priority
                className="opening-stage__image"
                sizes="100vw"
              />
            </div>
            <div className="opening-stage__layout">
              <div className="opening-stage__zone opening-stage__zone--left" aria-hidden="true" />
              <div className="opening-stage__zone opening-stage__zone--center" aria-hidden="true" />
              <div className="opening-stage__zone opening-stage__zone--right">
                <div className="opening-stage__cta stack opening-stage__cta-stack">
                  <button className="button button--primary" type="button" onClick={handleOpeningContinue}>
                    Begin Assessment
                  </button>
                </div>
              </div>
            </div>
          </section>
        ) : stage === "identity" || stage === "duplicate" ? (
          <>
            <section className="hero hero--compact">
              <span className="eyebrow">Details</span>
              <h1>Enter details before the questionnaire begins for proper assessment.</h1>
              <p>Your data would be kept private.</p>
            </section>

            <section className="identity-screen">
              <div className="panel stack panel--identity">
                {stage === "identity" && (
                  <>
                    <div className="stack">
                      <h2 className="section-title">Present details</h2>
                    </div>
                    <form className="form-grid" onSubmit={handleIdentitySubmit}>
                      <div className="field-grid field-grid--double">
                        <div className="field">
                          <label htmlFor="firstName">First Name</label>
                          <input className="input" id="firstName" value={identityForm.firstName} onChange={(event) => updateField("firstName", event.target.value)} />
                          {errors.firstName ? <p className="error-text">{errors.firstName}</p> : null}
                        </div>
                        <div className="field">
                          <label htmlFor="middleName">Middle Name</label>
                          <input className="input" id="middleName" value={identityForm.middleName} onChange={(event) => updateField("middleName", event.target.value)} />
                          {errors.middleName ? <p className="error-text">{errors.middleName}</p> : null}
                        </div>
                      </div>
                      <div className="field-grid field-grid--double">
                        <div className="field">
                          <label htmlFor="lastName">Last Name</label>
                          <input className="input" id="lastName" value={identityForm.lastName} onChange={(event) => updateField("lastName", event.target.value)} />
                          {errors.lastName ? <p className="error-text">{errors.lastName}</p> : null}
                        </div>
                        <div className="field">
                          <label htmlFor="dateOfBirth">Date of Birth</label>
                          <input className="input" id="dateOfBirth" type="date" placeholder="YYYY-MM-DD" value={identityForm.dateOfBirth} onChange={(event) => updateField("dateOfBirth", event.target.value)} />
                          {derivedAge !== null ? <p className="field__hint">Derived age: {derivedAge}</p> : null}
                          {errors.dateOfBirth ? <p className="error-text">{errors.dateOfBirth}</p> : null}
                        </div>
                      </div>
                      <div className="field">
                        <label htmlFor="location">Present location</label>
                        <input className="input" id="location" value={identityForm.location} onChange={(event) => updateField("location", event.target.value)} />
                        {errors.location ? <p className="error-text">{errors.location}</p> : null}
                      </div>
                      <div className="field-grid field-grid--double">
                        <div className="field">
                          <label htmlFor="email">Email address</label>
                          <input className="input" id="email" type="email" value={identityForm.email} onChange={(event) => updateField("email", event.target.value)} />
                          {errors.email ? <p className="error-text">{errors.email}</p> : null}
                        </div>
                        <div className="field">
                          <label htmlFor="countryCode">Country code</label>
                          <select className="input" id="countryCode" value={identityForm.countryCode} onChange={(event) => updateField("countryCode", event.target.value)}>
                            {countryPhoneOptions.map((country) => (
                              <option key={country.code} value={country.code}>
                                {country.name} ({country.dialCode})
                              </option>
                            ))}
                          </select>
                          {errors.countryCode ? <p className="error-text">{errors.countryCode}</p> : null}
                        </div>
                      </div>
                      <div className="field">
                        <label htmlFor="localPhoneNumber">Phone Number</label>
                        <input className="input" id="localPhoneNumber" type="tel" inputMode="numeric" value={identityForm.localPhoneNumber} onChange={(event) => updateField("localPhoneNumber", event.target.value)} />
                        {errors.localPhoneNumber ? <p className="error-text">{errors.localPhoneNumber}</p> : null}
                      </div>
                      <div className="button-row">
                        <button className="button button--primary" type="submit">Begin with questionnaire</button>
                      </div>
                    </form>
                  </>
                )}

                {stage === "duplicate" && (
                  <div className="stack">
                    <h2 className="section-title">This assessment is already on record</h2>
                    <p className="muted">{duplicateMessage}</p>
                    <div className="button-row">
                      <button className="button button--secondary" type="button" onClick={() => goToStage("identity")}>Review details</button>
                    </div>
                  </div>
                )}

                {uiError ? <p className="error-text">{uiError}</p> : null}
              </div>
            </section>
          </>
        ) : (
          <>
            <section className="hero hero--compact">
              <span className="eyebrow">Ayurvedic VPK Assessment</span>
              <h1 className="hero__single-line">Proceed with clarity and complete the assessment calmly.</h1>
              <p>VPK assessment is already registered for {registrantName}. Continue through instructions, the guided questionnaire, and the final result view.</p>
            </section>

            <section className={`split ${stage === "assessment" ? "split--assessment" : ""}`}>
              <div className="panel stack">
                {stage === "instructions" && (
                  <div className="stack">
                    <h2 className="section-title">Instructions</h2>
                    <div className="status-card">
                      <p className="muted">{instructionsText || "Loading instructions..."}</p>
                    </div>
                    {accessWindowExpiresAt ? (
                      <div className="status-card">
                        <p className="muted">This test window stays open until {new Date(accessWindowExpiresAt).toLocaleString()}.</p>
                      </div>
                    ) : null}
                    {protectionsActive ? (
                      <div className="status-card status-card--private">
                        <p className="muted">Private assessment mode is active for {account.displayName}. Copy, print, and context actions are disabled in this browser session.</p>
                      </div>
                    ) : null}
                    <div className="button-row">
                      <button className="button button--primary" type="button" onClick={handleAcknowledge}>
                        I have read and understood
                      </button>
                    </div>
                  </div>
                )}

                {stage === "start" && (
                  <div className="stack">
                    <h2 className="section-title">Start the assessment</h2>
                    <p className="muted">
                      You will answer one category at a time. Each screen requires one choice for Lifetime and one choice for Present.
                    </p>
                    <p className="muted">
                      If none of the options fits perfectly, choose the option that is the closest match.
                    </p>
                    <div className="status-card">
                      <p className="muted">Your acknowledgement has been recorded. Results remain hidden until the final step.</p>
                    </div>
                    {accessWindowExpiresAt ? (
                      <div className="status-card">
                        <p className="muted">This attempt must be completed before {new Date(accessWindowExpiresAt).toLocaleString()}.</p>
                      </div>
                    ) : null}
                    <div className="button-row">
                      <button className="button button--primary" type="button" onClick={() => void enterAssessment()}>
                        Start Test
                      </button>
                    </div>
                  </div>
                )}

                {stage === "assessment" && questionLoading && (
                  <div className="stack">
                    <div className="status-card">
                      <p className="muted">Loading the first question...</p>
                    </div>
                  </div>
                )}

                {stage === "assessment" && question && (
                  <div className="stack assessment-stage">
                    <div className="question-header">
                      <span className="question-counter">Question {question.index} of {question.total}</span>
                      <h2 className="question-title">{question.category.title}</h2>
                      {question.category.note ? <p className="muted">{question.category.note}</p> : null}
                    </div>
                    <div className="tracks">
                      <div className="track-card">
                        <p className="group-label">{question.lifetimeLabel}</p>
                        <div className="option-list" role="radiogroup" aria-label={question.lifetimeLabel}>
                          {question.options.map((option) => (
                            <button
                              key={option.id}
                              type="button"
                              className={`option-card ${lifetimeSelection === option.id ? "option-card--selected" : ""}`}
                              onClick={() => setLifetimeSelection(option.id)}
                              aria-pressed={lifetimeSelection === option.id}
                            >
                              <span className="option-card__title">{option.text}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                      <div className="track-card">
                        <p className="group-label">{question.presentLabel}</p>
                        <div className="option-list" role="radiogroup" aria-label={question.presentLabel}>
                          {question.options.map((option) => (
                            <button
                              key={option.id}
                              type="button"
                              className={`option-card ${presentSelection === option.id ? "option-card--selected" : ""}`}
                              onClick={() => setPresentSelection(option.id)}
                              aria-pressed={presentSelection === option.id}
                            >
                              <span className="option-card__title">{option.text}</span>
                            </button>
                          ))}
                        </div>
                      </div>
                    </div>
                    <div className="button-row">
                      <button className="button button--ghost" type="button" onClick={handleBack} disabled={question.index === 1}>Back</button>
                      <button className="button button--primary" type="button" onClick={handleNext}>
                        {question.index === question.total ? "Show Result" : "Next Question"}
                      </button>
                    </div>
                  </div>
                )}

                {stage === "result" && result && (
                  <div className="stack">
                    <div className="stack">
                      <h2 className="section-title">Final constitution view</h2>
                      <p className="muted">Your constitution profile is revealed only after all categories are submitted.</p>
                    </div>
                    <div className="result-grid">
                      <div className="result-track">
                        <span className="eyebrow">Lifetime / Prakriti</span>
                        <div className="constitution-pill">Constitution: {result.lifetime.constitutionLabel}</div>
                        <PieResultsChart data={result.charts.lifetime} title="Lifetime" />
                      </div>
                      <div className="result-track">
                        <span className="eyebrow">Present / Vikriti</span>
                        <div className="constitution-pill">Constitution: {result.present.constitutionLabel}</div>
                        <PieResultsChart data={result.charts.present} title="Present" />
                      </div>
                    </div>
                  </div>
                )}

                {uiError ? <p className="error-text">{uiError}</p> : null}
              </div>
            </section>
          </>
        )}
      </div>
    </main>
  );
}

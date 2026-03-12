"use client";

import { FormEvent, useEffect, useState } from "react";

import { PieResultsChart } from "@/components/PieResultsChart";
import type { AttemptSnapshot, QuestionPayload, ResultPayload, WizardStage } from "@/lib/types";
import { countryPhoneOptions, deriveAgeFromDateOfBirth } from "@/lib/validation";

type AssessmentAppProps = {
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

export function AssessmentApp({
  initialSnapshot,
  initialQuestion,
  initialResult,
}: AssessmentAppProps) {
  const [stage, setStage] = useState<WizardStage>(initialSnapshot?.stage ?? "identity");
  const [attemptId, setAttemptId] = useState<string | null>(initialSnapshot?.attemptId ?? initialResult?.attemptId ?? null);
  const [questionIndex, setQuestionIndex] = useState(initialSnapshot?.questionIndex ?? 1);
  const [identityForm, setIdentityForm] = useState(emptyForm);
  const [instructionsText, setInstructionsText] = useState("");
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [uiError, setUiError] = useState("");
  const [duplicateMessage, setDuplicateMessage] = useState("");
  const [question, setQuestion] = useState<QuestionPayload | null>(initialQuestion);
  const [lifetimeSelection, setLifetimeSelection] = useState<string | null>(
    initialQuestion?.savedResponse.lifetimeOptionId ?? null,
  );
  const [presentSelection, setPresentSelection] = useState<string | null>(
    initialQuestion?.savedResponse.presentOptionId ?? null,
  );
  const [result, setResult] = useState<ResultPayload | null>(initialResult);
  const derivedAge = identityForm.dateOfBirth
    ? deriveAgeFromDateOfBirth(identityForm.dateOfBirth)
    : null;

  function updateField<Key extends keyof IdentityForm>(key: Key, value: IdentityForm[Key]) {
    setIdentityForm((current) => ({ ...current, [key]: value }));
  }

  async function loadQuestion(index: number) {
    setUiError("");
    const response = await fetch(`/api/questions/${index}`);
    const payload = await response.json();

    if (!response.ok) {
      setUiError(payload.error ?? "Unable to load this question.");
      return;
    }

    setQuestion(payload);
    setQuestionIndex(payload.index);
    setLifetimeSelection(payload.savedResponse.lifetimeOptionId);
    setPresentSelection(payload.savedResponse.presentOptionId);
  }

  useEffect(() => {
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
  }, [instructionsText, stage]);

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
        setStage("duplicate");
        return;
      }

      setUiError(payload.error ?? "Unable to save your details.");
      return;
    }

    setAttemptId(payload.attemptId);
    setStage("instructions");
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

    setStage("start");
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
    setStage("result");
  }

  return (
    <main className="app-shell">
      <div className="app-shell__inner">
        <section className="hero">
          <span className="eyebrow">Ayurvedic VPK Assessment</span>
          <h1>Reflective constitution mapping, one quiet step at a time.</h1>
          <p>
            Identity is recorded first, instructions are acknowledged explicitly, and every category is answered in two parallel tracks before results appear at the end.
          </p>
        </section>

        <section className="split">
          <div className="panel stack">
            {stage === "identity" && (
              <>
                <div className="stack">
                  <h2 className="section-title">Begin with your present details</h2>
                  <p className="muted">
                    This one-time assessment requires your present identity details before any questionnaire access is allowed.
                  </p>
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
                      {derivedAge !== null ? <p className="field__hint">Derived age: {derivedAge}</p> : <p className="field__hint">Use the calendar or type a valid date in YYYY-MM-DD format.</p>}
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
                    <label htmlFor="localPhoneNumber">Local phone number</label>
                    <input className="input" id="localPhoneNumber" type="tel" inputMode="numeric" value={identityForm.localPhoneNumber} onChange={(event) => updateField("localPhoneNumber", event.target.value)} />
                    {identityForm.countryCode === "IN" ? <p className="field__hint">For India, enter exactly 10 digits.</p> : null}
                    {errors.localPhoneNumber ? <p className="error-text">{errors.localPhoneNumber}</p> : null}
                  </div>
                  <div className="button-row">
                    <button className="button button--primary" type="submit">Save details and continue</button>
                  </div>
                </form>
              </>
            )}

            {stage === "duplicate" && (
              <div className="stack">
                <h2 className="section-title">This assessment is already on record</h2>
                <p className="muted">{duplicateMessage}</p>
                <div className="button-row">
                  <button className="button button--secondary" type="button" onClick={() => setStage("identity")}>Review details</button>
                </div>
              </div>
            )}

            {stage === "instructions" && (
              <div className="stack">
                <h2 className="section-title">Instructions</h2>
                <div className="status-card">
                  <p className="muted">{instructionsText || "Loading instructions..."}</p>
                </div>
                <div className="button-row">
                  <button className="button button--primary" type="button" onClick={handleAcknowledge}>
                    I have read and understood
                  </button>
                </div>
              </div>
            )}

            {stage === "start" && (
              <div className="stack">
                <h2 className="section-title">Start the 40-category assessment</h2>
                <p className="muted">
                  You will answer one category at a time. Each screen requires one choice for Lifetime and one choice for Present.
                </p>
                <div className="status-card">
                  <p className="muted">Your acknowledgement has been recorded. Results remain hidden until the final step.</p>
                </div>
                <div className="button-row">
                  <button className="button button--primary" type="button" onClick={() => setStage("assessment")}>
                    Start Test
                  </button>
                </div>
              </div>
            )}

            {stage === "assessment" && question && (
              <div className="stack">
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
                          <span className="option-card__meta">{lifetimeSelection === option.id ? "Selected for Lifetime" : "Choose for Lifetime"}</span>
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
                          <span className="option-card__meta">{presentSelection === option.id ? "Selected for Present" : "Choose for Present"}</span>
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
                  <p className="muted">These totals are generated only after the last question has been submitted.</p>
                </div>
                <div className="result-grid">
                  <div className="result-track">
                    <span className="eyebrow">Lifetime / Prakriti</span>
                    <div className="constitution-pill">Constitution: {result.lifetime.constitutionLabel}</div>
                    <PieResultsChart data={result.charts.lifetime} title="Lifetime" />
                    <div className="totals-list">
                      <div className="totals-item"><span>V total</span><strong>{result.lifetime.V}</strong></div>
                      <div className="totals-item"><span>P total</span><strong>{result.lifetime.P}</strong></div>
                      <div className="totals-item"><span>K total</span><strong>{result.lifetime.K}</strong></div>
                    </div>
                  </div>
                  <div className="result-track">
                    <span className="eyebrow">Present / Vikriti</span>
                    <div className="constitution-pill">Constitution: {result.present.constitutionLabel}</div>
                    <PieResultsChart data={result.charts.present} title="Present" />
                    <div className="totals-list">
                      <div className="totals-item"><span>V total</span><strong>{result.present.V}</strong></div>
                      <div className="totals-item"><span>P total</span><strong>{result.present.P}</strong></div>
                      <div className="totals-item"><span>K total</span><strong>{result.present.K}</strong></div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {uiError ? <p className="error-text">{uiError}</p> : null}
          </div>

          <aside className="panel panel--dense stack">
            <div className="badge-row">
              <span className="badge">One-time access</span>
              <span className="badge">40 categories</span>
              <span className="badge">Two parallel tracks</span>
            </div>
            <div className="divider" />
            <div className="status-card">
              <p className="muted">
                The app records identity first, captures instruction acknowledgement with timestamp, then guides the user through one category at a time without revealing any interim score.
              </p>
            </div>
          </aside>
        </section>
      </div>
    </main>
  );
}

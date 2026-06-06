import { Component, input, signal } from '@angular/core';
import { CommonModule, KeyValuePipe } from '@angular/common';
import { MatIconModule } from '@angular/material/icon';

export interface HelixResult {
  crypto_verified: boolean;
  hospital_synced: boolean;
  failure_reason?: string | null;
  message?: string | null;
  label: number | null;
  label_name: string | null;
  confidences: Record<string, number> | null;
  plaintext_matches: boolean;
  ciphertext_sample: number[];
  ciphertext_size_bytes: number;
  timing: {
    encrypt_ms: number;
    compute_ms: number;
    decrypt_ms: number;
    total_ms: number;
  };
}

@Component({
  selector: 'app-helix-result-card',
  standalone: true,
  imports: [CommonModule, KeyValuePipe, MatIconModule],
  template: `
    <div
      class="helix-panel-wrap"
      (click)="flipped.set(!flipped())"
      (keydown.enter)="flipped.set(!flipped())"
      (keydown.space)="flipped.set(!flipped()); $event.preventDefault()"
      tabindex="0"
      role="button"
      [attr.aria-label]="flipped() ? 'Show routing result' : 'Show encrypted ciphertext'">
      <div class="helix-stack">
        <div class="helix-panel helix-front" [class.helix-panel-hidden]="flipped()" [attr.aria-hidden]="flipped()">
          <span class="helix-toggle-hint"><mat-icon>flip</mat-icon> tap for ciphertext</span>
          <div class="helix-header">
            <span class="helix-badge">HELIX</span>
            <span class="helix-label helix-label-route">
              {{ practitionerShort() }} asked → {{ specialistShort() }} routed (encrypted)
            </span>
          </div>
          @if (result(); as r) {
            @if (r.crypto_verified && r.label_name && r.confidences) {
              <div class="helix-department">{{ r.label_name.replace('_', ' ') }}</div>
              <div class="helix-confidences">
                @for (entry of r.confidences | keyvalue; track entry.key) {
                  <div class="conf-row">
                    <span class="conf-name">{{ entry.key.replace('_', ' ') }}</span>
                    <div class="conf-bar-bg">
                      <div class="conf-bar" [style.width.%]="entry.value * 100"></div>
                    </div>
                    <span class="conf-pct">{{ (entry.value * 100).toFixed(0) }}%</span>
                  </div>
                }
              </div>
            } @else {
              <div class="helix-failure">
                <mat-icon class="helix-failure-icon">gpp_bad</mat-icon>
                <p class="helix-failure-title">Routing not verified</p>
                <p class="helix-failure-message">{{ r.message ?? 'HELIX key mismatch — result cannot be trusted.' }}</p>
                <p class="helix-failure-hint">
                  Only the clinic can open the encrypted routing result. Without a matching HELIX key
                  shared with the hospital, no department assignment is shown.
                </p>
              </div>
            }
            <div class="helix-footer">
              @if (r.crypto_verified) {
                <span class="helix-check helix-check-ok">Crypto verified</span>
              } @else {
                <span class="helix-check helix-check-fail">Crypto not verified</span>
              }
              <span class="helix-timing">{{ r.timing.total_ms.toFixed(0) }}ms</span>
            </div>
          }
        </div>
        <div class="helix-panel helix-back" [class.helix-panel-hidden]="!flipped()" [attr.aria-hidden]="!flipped()">
          <span class="helix-toggle-hint"><mat-icon>flip</mat-icon> tap for result</span>
          <div class="helix-flow">
            <section class="helix-flow-section">
              <h4 class="helix-flow-title">{{ practitionerShort() }} (your clinic)</h4>
              <p class="helix-flow-desc">Encoded the query locally, encrypted with CKKS, holds the secret key.</p>
              <p class="helix-flow-note">Secret key: held locally (never sent)</p>
            </section>
            <div class="helix-flow-divider" aria-hidden="true">
              <mat-icon>arrow_downward</mat-icon>
              <span>encrypted data</span>
            </div>
            <section class="helix-flow-section">
              <h4 class="helix-flow-title">{{ specialistShort() }} (specialist hub)</h4>
              <p class="helix-flow-desc">
                Computed on {{ formatCiphertextSize(result().ciphertext_size_bytes) }} of encrypted data without ever decrypting.
              </p>
              <div class="helix-heatmap-wrap">
                <div class="helix-heatmap" aria-hidden="true">
                  @for (v of result().ciphertext_sample; track $index) {
                    <div class="heat-cell" [style.background]="cellColor(v, result().ciphertext_sample)"></div>
                  }
                </div>
                <div class="helix-heatmap-legend" aria-hidden="true">
                  <div class="legend-bar"></div>
                  <span>Low</span>
                  <span>High</span>
                </div>
              </div>
              <p class="helix-ciphertext-meta">
                {{ formatCoefficientCount(result().ciphertext_size_bytes) }} CKKS coefficients
              </p>
              <p class="helix-flow-note">No secret key. Cannot decrypt. Only performed blind matrix multiply.</p>
            </section>
          </div>
        </div>
      </div>
    </div>
  `,
  styles: `
    :host {
      display: block;
      width: 100%;
      max-width: 340px;
    }

    .helix-panel-wrap {
      cursor: pointer;
      outline: none;
      width: 100%;
      border-radius: 10px;
      overflow: hidden;
      border: 1px solid #c4b5fd;
      background: #faf5ff;
    }

    .helix-panel-wrap:focus-visible {
      box-shadow: inset 0 0 0 2px #7c3aed;
    }

    .helix-stack {
      display: grid;
      width: 100%;
    }

    .helix-stack > .helix-panel {
      grid-area: 1 / 1;
      width: 100%;
      min-height: 100%;
      box-sizing: border-box;
    }

    .helix-panel-hidden {
      visibility: hidden;
      pointer-events: none;
    }

    .helix-panel {
      padding: 16px;
    }

    .helix-front {
      background: #faf5ff;
      color: #1e293b;
    }

    .helix-back {
      background: #1e1b4b;
      color: #e9d5ff;
    }

    .helix-toggle-hint {
      display: flex;
      align-items: center;
      justify-content: flex-end;
      gap: 4px;
      margin: -4px 0 8px;
      font-size: 0.65rem;
      color: #7c3aed;
      opacity: 0.85;
    }

    .helix-toggle-hint mat-icon {
      font-size: 16px;
      width: 16px;
      height: 16px;
    }

    .helix-back .helix-toggle-hint {
      color: #c4b5fd;
    }

    .helix-header {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-bottom: 12px;
    }

    .helix-badge {
      background: #7c3aed;
      color: white;
      font-size: 0.65rem;
      font-weight: 700;
      padding: 2px 8px;
      border-radius: 4px;
      letter-spacing: 0.05em;
    }

    .helix-label {
      font-size: 0.8rem;
      color: #6b7280;
    }

    .helix-label-route {
      flex: 1;
      line-height: 1.35;
    }

    .helix-department {
      font-size: 1.3rem;
      font-weight: 600;
      color: #1e1b4b;
      margin-bottom: 14px;
    }

    .helix-confidences {
      display: flex;
      flex-direction: column;
      gap: 6px;
      margin-bottom: 12px;
    }

    .conf-row {
      display: flex;
      align-items: center;
      gap: 8px;
    }

    .conf-name {
      width: 110px;
      font-size: 0.75rem;
      color: #4b5563;
      flex-shrink: 0;
    }

    .conf-bar-bg {
      flex: 1;
      height: 6px;
      background: #e5e7eb;
      border-radius: 3px;
      overflow: hidden;
    }

    .conf-bar {
      height: 100%;
      background: #7c3aed;
      border-radius: 3px;
      transition: width 0.3s ease;
    }

    .conf-pct {
      width: 36px;
      font-size: 0.7rem;
      color: #6b7280;
      text-align: right;
    }

    .helix-footer {
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding-top: 10px;
      border-top: 1px solid #e9d5ff;
      font-size: 0.7rem;
      color: #6b7280;
    }

    .helix-check {
      font-weight: 600;
    }

    .helix-check-ok {
      color: #059669;
    }

    .helix-check-fail {
      color: #dc2626;
    }

    .helix-failure {
      margin-bottom: 12px;
      padding: 12px;
      border-radius: 8px;
      background: #fef2f2;
      border: 1px solid #fecaca;
    }

    .helix-failure-icon {
      color: #dc2626;
      font-size: 28px;
      width: 28px;
      height: 28px;
      margin-bottom: 6px;
    }

    .helix-failure-title {
      margin: 0 0 6px;
      font-size: 1rem;
      font-weight: 600;
      color: #991b1b;
    }

    .helix-failure-message {
      margin: 0 0 8px;
      font-size: 0.78rem;
      line-height: 1.45;
      color: #7f1d1d;
    }

    .helix-failure-hint {
      margin: 0;
      font-size: 0.68rem;
      line-height: 1.4;
      color: #b91c1c;
      font-style: italic;
    }

    .helix-flow {
      display: flex;
      flex-direction: column;
      gap: 0;
    }

    .helix-flow-title {
      margin: 0 0 6px;
      font-size: 0.78rem;
      font-weight: 600;
      color: #e9d5ff;
      line-height: 1.3;
    }

    .helix-flow-desc {
      margin: 0 0 6px;
      font-size: 0.7rem;
      line-height: 1.4;
      color: #c4b5fd;
    }

    .helix-flow-note {
      margin: 0;
      font-size: 0.65rem;
      line-height: 1.35;
      color: #a78bfa;
      font-style: italic;
    }

    .helix-flow-divider {
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 2px;
      margin: 10px 0;
      padding: 6px 0;
      border-top: 1px dashed #4c1d95;
      border-bottom: 1px dashed #4c1d95;
      color: #a78bfa;
      font-size: 0.6rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }

    .helix-flow-divider mat-icon {
      font-size: 18px;
      width: 18px;
      height: 18px;
    }

    .helix-heatmap-wrap {
      display: flex;
      gap: 8px;
      align-items: stretch;
      margin: 8px 0 6px;
    }

    .helix-heatmap {
      display: grid;
      grid-template-columns: repeat(8, 1fr);
      gap: 1px;
      flex: 1;
      max-width: 160px;
      aspect-ratio: 1;
      border-radius: 4px;
      overflow: hidden;
      border: 1px solid #4c1d95;
    }

    .heat-cell {
      aspect-ratio: 1;
    }

    .helix-heatmap-legend {
      display: flex;
      flex-direction: column;
      align-items: center;
      width: 12px;
      font-size: 0.5rem;
      color: #a78bfa;
      gap: 2px;
    }

    .legend-bar {
      flex: 1;
      width: 8px;
      border-radius: 3px;
      background: linear-gradient(to bottom, #d94a1e, #e09040, #f0e060, #60c8a0, #1e6eb4);
    }

    .helix-ciphertext-meta {
      margin: 0;
      font-size: 0.68rem;
      line-height: 1.4;
      color: #a78bfa;
      font-family: ui-monospace, monospace;
      word-break: break-word;
    }
  `,
})
export class HelixResultCardComponent {
  result = input.required<HelixResult>();
  practitionerShort = input('Clinic');
  specialistShort = input('Hospital');

  flipped = signal(false);

  formatCiphertextSize(bytes: number): string {
    if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
    if (bytes >= 1000) return `${(bytes / 1000).toFixed(1)} KB`;
    return `${bytes} B`;
  }

  formatCoefficientCount(bytes: number): string {
    const n = Math.max(1, Math.round(bytes / 8));
    return n >= 1000 ? `${(n / 1000).toFixed(1)}k` : String(n);
  }

  cellColor(value: number, sample: number[]): string {
    const min = Math.min(...sample);
    const max = Math.max(...sample);
    const t = max > min ? (value - min) / (max - min) : 0.5;
    const stops = [
      [30, 110, 180],
      [96, 200, 160],
      [240, 224, 96],
      [224, 144, 64],
      [217, 74, 30],
    ];
    const idx = t * (stops.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.min(lo + 1, stops.length - 1);
    const f = idx - lo;
    const r = Math.round(stops[lo][0] + f * (stops[hi][0] - stops[lo][0]));
    const g = Math.round(stops[lo][1] + f * (stops[hi][1] - stops[lo][1]));
    const b = Math.round(stops[lo][2] + f * (stops[hi][2] - stops[lo][2]));
    return `rgb(${r},${g},${b})`;
  }
}

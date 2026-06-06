import { Component, ElementRef, inject, OnDestroy, OnInit, signal, viewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatIconModule } from '@angular/material/icon';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatButtonToggleModule } from '@angular/material/button-toggle';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { HelixResult, HelixResultCardComponent } from './helix-result-card.component';

type DemoMode = 'sealed' | 'helix';

interface WireOut {
  url: string;
  method: string;
  step: number;
  bytes_on_wire: number;
  vector_b64_prefix: string;
}

interface PacketEvent {
  type: string;
  step: number;
  encrypted_vector_sample: number[];
  tok_per_sec: number;
  wire_out: WireOut;
  sent_at: string;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  text?: string;
  streaming?: boolean;
  helix?: HelixResult;
}

interface ActiveLabels {
  practitioner: string;
  specialist: string;
  practitioner_short: string;
  specialist_short: string;
}

interface AppConfig {
  ready?: boolean;
  helix_available?: boolean;
  helix_bootstrapped?: boolean;
  helix_key_fingerprint?: string | null;
  helix_hospital_fingerprint?: string | null;
  helix_hospital_synced?: boolean;
  active_labels?: ActiveLabels;
}

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    MatToolbarModule,
    MatIconModule,
    MatFormFieldModule,
    MatInputModule,
    MatButtonModule,
    MatButtonToggleModule,
    MatProgressSpinnerModule,
    HelixResultCardComponent,
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent implements OnInit, OnDestroy {
  protected readonly Math = Math;
  private http = inject(HttpClient);
  private ws: WebSocket | null = null;
  private configPoll: ReturnType<typeof setInterval> | null = null;

  scrollContainer = viewChild<ElementRef>('scrollContainer');
  packetLogContainer = viewChild<ElementRef>('packetLogContainer');

  mode = signal<DemoMode>('sealed');
  helixAvailable = signal(false);

  passphrase = signal('');
  keyFingerprint = signal<string | null>(null);
  keyLoading = signal(false);

  helixKey = signal('');
  helixKeyFingerprint = signal<string | null>(null);
  helixHospitalFingerprint = signal<string | null>(null);
  helixHospitalSynced = signal(false);
  helixKeyLoading = signal(false);
  helixSyncLoading = signal(false);

  ready = signal(false);
  generating = signal(false);
  phrases = signal<string[]>([]);
  showPhrases = signal(false);

  messages = signal<ChatMessage[]>([]);
  draft = '';

  lastWire = signal<WireOut | null>(null);
  lastVectorSample = signal<number[]>([]);
  packetLog = signal<PacketEvent[]>([]);
  currentStep = signal(0);
  modelLabels = signal<ActiveLabels | null>(null);

  ngOnInit(): void {
    this.http.get<{ phrases: string[] }>('/api/phrases').subscribe({
      next: (r) => this.phrases.set(r.phrases),
    });
    this.refreshConfig();
    this.configPoll = setInterval(() => this.refreshConfig(), 3000);
  }

  ngOnDestroy(): void {
    this.ws?.close();
    if (this.configPoll) clearInterval(this.configPoll);
  }

  private refreshConfig(): void {
    this.http.get<AppConfig>('/api/config').subscribe({
      next: (r) => {
        this.ready.set(!!r.ready);
        this.helixAvailable.set(!!r.helix_available);
        if (r.active_labels) {
          this.modelLabels.set(r.active_labels);
        }
        this.helixKeyFingerprint.set(
          r.helix_bootstrapped && r.helix_key_fingerprint ? r.helix_key_fingerprint : null,
        );
        this.helixHospitalFingerprint.set(r.helix_hospital_fingerprint ?? null);
        this.helixHospitalSynced.set(!!r.helix_hospital_synced);
      },
    });
  }

  onModeChange(next: DemoMode): void {
    const prev = this.mode();
    this.mode.set(next);
    if (prev === 'helix' && next === 'sealed') {
      this.http.post('/api/session/helix-key/clear', {}).subscribe({
        next: () => {
          this.helixKeyFingerprint.set(null);
          this.refreshConfig();
        },
        error: () => this.helixKeyFingerprint.set(null),
      });
    }
  }

  applyPassphrase(): void {
    const p = this.passphrase().trim();
    if (!p) return;
    this.keyLoading.set(true);
    this.http.post<{ fingerprint: string }>('/api/session/key', { passphrase: p }).subscribe({
      next: (r) => {
        this.keyFingerprint.set(r.fingerprint);
        this.keyLoading.set(false);
      },
      error: () => this.keyLoading.set(false),
    });
  }

  applyHelixKey(): void {
    const k = this.helixKey().trim();
    if (!k) return;
    this.helixKeyLoading.set(true);
    this.http
      .post<{
        fingerprint: string;
        hospital_synced: boolean;
        hospital_fingerprint: string | null;
      }>('/api/session/helix-key', { helix_key: k })
      .subscribe({
        next: (r) => {
          this.helixKeyFingerprint.set(r.fingerprint);
          this.helixHospitalFingerprint.set(r.hospital_fingerprint ?? null);
          this.helixHospitalSynced.set(!!r.hospital_synced);
          this.helixKeyLoading.set(false);
        },
        error: () => this.helixKeyLoading.set(false),
      });
  }

  syncHelixToHospital(): void {
    this.helixSyncLoading.set(true);
    this.http
      .post<{
        fingerprint: string;
        hospital_synced: boolean;
        hospital_fingerprint: string | null;
      }>('/api/session/helix-key/sync', {})
      .subscribe({
        next: (r) => {
          this.helixKeyFingerprint.set(r.fingerprint);
          this.helixHospitalFingerprint.set(r.hospital_fingerprint ?? null);
          this.helixHospitalSynced.set(!!r.hospital_synced);
          this.helixSyncLoading.set(false);
        },
        error: () => this.helixSyncLoading.set(false),
      });
  }

  canSubmit(): boolean {
    if (!this.keyFingerprint() || this.generating()) return false;
    if (this.mode() === 'helix') {
      return !!this.helixKeyFingerprint() && !!this.draft.trim();
    }
    return !!this.draft.trim();
  }

  formatTime(iso: string): string {
    try {
      return new Date(iso).toLocaleTimeString();
    } catch {
      return iso;
    }
  }

  packetLine(pkt: PacketEvent): string {
    const kb = (pkt.wire_out.bytes_on_wire / 1024).toFixed(1);
    return `[${this.formatTime(pkt.sent_at)}] Packet ${pkt.step} sent — ${kb} KB ciphertext`;
  }

  selectPhrase(phrase: string): void {
    this.draft = phrase;
    this.showPhrases.set(false);
    this.submit();
  }

  submit(): void {
    const prompt = this.draft.trim();
    if (!prompt || !this.canSubmit()) return;

    if (this.mode() === 'helix') {
      this.submitHelix(prompt);
      return;
    }
    this.submitSealed(prompt);
  }

  private submitHelix(prompt: string): void {
    this.messages.update((m) => [...m, { role: 'user', text: prompt }]);
    this.messages.update((m) => [
      ...m,
      { role: 'assistant', text: 'Running HELIX encrypted routing…', streaming: true },
    ]);
    this.draft = '';
    this.generating.set(true);

    this.http.post<HelixResult>('/api/helix/classify', { prompt }).subscribe({
      next: (result) => {
        this.generating.set(false);
        this.messages.update((msgs) => {
          const copy = [...msgs];
          const last = copy[copy.length - 1];
          if (last?.role === 'assistant') {
            copy[copy.length - 1] = { role: 'assistant', helix: result };
          }
          return copy;
        });
        this.scrollToBottom();
      },
      error: (err) => {
        this.generating.set(false);
        const msg = err?.error?.detail ?? 'HELIX classification failed';
        this.messages.update((msgs) => {
          const copy = [...msgs];
          const last = copy[copy.length - 1];
          if (last?.role === 'assistant') {
            copy[copy.length - 1] = { role: 'assistant', text: `Error: ${msg}`, streaming: false };
          }
          return copy;
        });
      },
    });
  }

  private submitSealed(prompt: string): void {
    this.messages.update((m) => [...m, { role: 'user', text: prompt }]);
    this.messages.update((m) => [
      ...m,
      { role: 'assistant', text: 'Sending encrypted packets…', streaming: true },
    ]);
    this.draft = '';
    this.packetLog.set([]);
    this.currentStep.set(0);
    this.generating.set(true);

    const ws = this.connectWs();
    const payload = JSON.stringify({ prompt });

    const send = () => ws.send(payload);
    if (ws.readyState === WebSocket.OPEN) {
      send();
    } else {
      ws.addEventListener('open', send, { once: true });
    }
  }

  private connectWs(): WebSocket {
    if (this.ws?.readyState === WebSocket.OPEN) return this.ws;
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    this.ws = new WebSocket(`${protocol}//${window.location.host}/ws/generate`);

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'packet') {
        const pkt = data as PacketEvent;
        this.currentStep.set(pkt.step);
        this.lastWire.set(pkt.wire_out ?? null);
        this.lastVectorSample.set(pkt.encrypted_vector_sample ?? []);
        this.packetLog.update((log) => [...log, pkt]);
        this.messages.update((msgs) => {
          const copy = [...msgs];
          const last = copy[copy.length - 1];
          if (last?.role === 'assistant' && last.streaming) {
            copy[copy.length - 1] = {
              ...last,
              text: `Sending encrypted packets… (step ${pkt.step + 1})`,
            };
          }
          return copy;
        });
        this.scrollPacketLog();
      } else if (data.type === 'done') {
        this.generating.set(false);
        this.messages.update((msgs) => {
          const copy = [...msgs];
          const last = copy[copy.length - 1];
          if (last?.role === 'assistant') {
            copy[copy.length - 1] = { ...last, text: data.full_text, streaming: false };
          }
          return copy;
        });
        this.scrollToBottom();
      } else if (data.type === 'error') {
        this.generating.set(false);
        this.messages.update((msgs) => {
          const copy = [...msgs];
          const last = copy[copy.length - 1];
          if (last?.role === 'assistant') {
            copy[copy.length - 1] = { role: 'assistant', text: `Error: ${data.message}`, streaming: false };
          }
          return copy;
        });
      }
    };

    this.ws.onerror = () => {
      this.generating.set(false);
    };

    return this.ws;
  }

  private scrollToBottom(): void {
    setTimeout(() => {
      const el = this.scrollContainer()?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
    }, 0);
  }

  private scrollPacketLog(): void {
    setTimeout(() => {
      const el = this.packetLogContainer()?.nativeElement;
      if (el) el.scrollTop = el.scrollHeight;
    }, 0);
  }
}

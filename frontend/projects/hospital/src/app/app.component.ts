import { Component, inject, OnDestroy, OnInit, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { MatToolbarModule } from '@angular/material/toolbar';
import { MatIconModule } from '@angular/material/icon';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatInputModule } from '@angular/material/input';
import { MatButtonModule } from '@angular/material/button';
import { MatProgressSpinnerModule } from '@angular/material/progress-spinner';
import { MatExpansionModule } from '@angular/material/expansion';
import { MatTabsModule } from '@angular/material/tabs';

interface PacketActivity {
  type: string;
  step: number;
  bytes_on_wire: number;
  vector_sample: number[];
  vector_b64_prefix?: string;
  received_at: string;
}

interface HelixActivity {
  type: string;
  bytes_on_wire: number;
  ciphertext_sample: number[];
  compute_ms: number;
  received_at: string;
}

interface PacketGroup {
  id: number;
  packets: PacketActivity[];
  expanded: boolean;
  startedAt: string;
}

interface HelixRequest {
  id: number;
  activity: HelixActivity;
  expanded: boolean;
}

interface ActiveLabels {
  practitioner: string;
  specialist: string;
  practitioner_short: string;
  specialist_short: string;
}

interface AppConfig {
  active_labels?: ActiveLabels;
  helix_available?: boolean;
  helix_bootstrapped?: boolean;
  helix_key_fingerprint?: string | null;
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
    MatProgressSpinnerModule,
    MatExpansionModule,
    MatTabsModule,
  ],
  templateUrl: './app.component.html',
  styleUrl: './app.component.scss',
})
export class AppComponent implements OnInit, OnDestroy {
  protected readonly Math = Math;
  private http = inject(HttpClient);
  private ws: WebSocket | null = null;
  private configPoll: ReturnType<typeof setInterval> | null = null;

  passphrase = signal('');
  keyFingerprint = signal<string | null>(null);
  keyLoading = signal(false);
  connected = signal(false);

  helixAvailable = signal(false);
  helixBootstrapped = signal(false);
  helixKeyFingerprint = signal<string | null>(null);

  activeTab = signal(0);

  lastPacket = signal<PacketActivity | null>(null);
  requestGroups = signal<PacketGroup[]>([]);

  lastHelix = signal<HelixActivity | null>(null);
  helixRequests = signal<HelixRequest[]>([]);

  modelLabels = signal<ActiveLabels | null>(null);

  ngOnInit(): void {
    this.refreshConfig();
    this.configPoll = setInterval(() => this.refreshConfig(), 3000);
    this.connectActivity();
  }

  ngOnDestroy(): void {
    this.ws?.close();
    if (this.configPoll) clearInterval(this.configPoll);
  }

  private refreshConfig(): void {
    this.http.get<AppConfig>('/api/config').subscribe({
      next: (r) => {
        if (r.active_labels) {
          this.modelLabels.set(r.active_labels);
        }
        this.helixAvailable.set(!!r.helix_available);
        this.helixBootstrapped.set(!!r.helix_bootstrapped);
        this.helixKeyFingerprint.set(r.helix_key_fingerprint ?? null);
      },
    });
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

  formatTime(iso: string): string {
    try {
      return new Date(iso).toLocaleTimeString();
    } catch {
      return iso;
    }
  }

  packetLine(pkt: PacketActivity): string {
    const kb = (pkt.bytes_on_wire / 1024).toFixed(1);
    return `[${this.formatTime(pkt.received_at)}] Packet ${pkt.step} — ${kb} KB ciphertext`;
  }

  helixLine(req: HelixRequest): string {
    const kb = (req.activity.bytes_on_wire / 1024).toFixed(1);
    return `${kb} KB CKKS blob · ${req.activity.compute_ms.toFixed(0)}ms compute`;
  }

  groupTitle(group: PacketGroup): string {
    return `Request ${group.id} · ${this.formatTime(group.startedAt)}`;
  }

  groupSummary(group: PacketGroup): string {
    const totalKb = group.packets.reduce((sum, p) => sum + p.bytes_on_wire, 0) / 1024;
    const n = group.packets.length;
    return `${n} packet${n === 1 ? '' : 's'} · ${totalKb.toFixed(1)} KB total`;
  }

  helixTitle(req: HelixRequest): string {
    return `HELIX request ${req.id} · ${this.formatTime(req.activity.received_at)}`;
  }

  onPanelExpanded(groupId: number, expanded: boolean): void {
    this.requestGroups.update((groups) =>
      groups.map((g) => (g.id === groupId ? { ...g, expanded } : g)),
    );
  }

  onHelixExpanded(reqId: number, expanded: boolean): void {
    this.helixRequests.update((reqs) =>
      reqs.map((r) => (r.id === reqId ? { ...r, expanded } : r)),
    );
  }

  formatCiphertextSize(bytes: number): string {
    if (bytes >= 1_000_000) return `${(bytes / 1_000_000).toFixed(1)} MB`;
    if (bytes >= 1000) return `${(bytes / 1000).toFixed(1)} KB`;
    return `${bytes} B`;
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

  private appendPacket(data: PacketActivity): void {
    this.lastPacket.set(data);
    this.activeTab.set(0);

    if (data.step === 0) {
      this.requestGroups.update((groups) => {
        const collapsed = groups.map((g) => ({ ...g, expanded: false }));
        return [
          ...collapsed,
          {
            id: groups.length + 1,
            packets: [data],
            expanded: true,
            startedAt: data.received_at,
          },
        ];
      });
      return;
    }

    this.requestGroups.update((groups) => {
      if (groups.length === 0) {
        return [
          {
            id: 1,
            packets: [data],
            expanded: true,
            startedAt: data.received_at,
          },
        ];
      }
      const copy = [...groups];
      const last = copy[copy.length - 1];
      copy[copy.length - 1] = {
        ...last,
        packets: [...last.packets, data],
      };
      return copy;
    });
  }

  private appendHelix(data: HelixActivity): void {
    this.lastHelix.set(data);
    this.activeTab.set(1);

    this.helixRequests.update((reqs) => {
      const collapsed = reqs.map((r) => ({ ...r, expanded: false }));
      return [
        ...collapsed,
        {
          id: reqs.length + 1,
          activity: data,
          expanded: true,
        },
      ];
    });
  }

  private connectActivity(): void {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    this.ws = new WebSocket(`${protocol}//${window.location.host}/ws/activity`);

    this.ws.onopen = () => this.connected.set(true);
    this.ws.onclose = () => {
      this.connected.set(false);
      setTimeout(() => this.connectActivity(), 2000);
    };

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'packet') {
        this.appendPacket(data as PacketActivity);
      } else if (data.type === 'helix_compute') {
        this.appendHelix(data as HelixActivity);
      }
    };
  }

  clearStream(): void {
    this.requestGroups.set([]);
    this.lastPacket.set(null);
    this.helixRequests.set([]);
    this.lastHelix.set(null);
  }
}

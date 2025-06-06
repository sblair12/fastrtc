<script lang="ts">
  import { Empty } from "@gradio/atoms";
  import { BlockLabel } from "@gradio/atoms";
  import { Music } from "@gradio/icons";
  import type { I18nFormatter } from "@gradio/utils";
  import { createEventDispatcher } from "svelte";
  import type { WebRTCValue } from "./utils";

  import { start, stop } from "./webrtc_utils";
  import AudioWave from "./AudioWave.svelte";

  export let value: string | WebRTCValue | null = null;
  export let label: string | undefined = undefined;
  export let show_label = true;
  export let rtc_configuration: Object | null = null;
  export let i18n: I18nFormatter;
  export let on_change_cb: (msg: "change" | "tick") => void;
  export let icon: string | undefined = undefined;
  export let icon_button_color: string = "var(--color-accent)";
  export let pulse_color: string = "var(--color-accent)";
  export let icon_radius: number = 50;

  export let server: {
    offer: (body: any) => Promise<any>;
    turn: () => Promise<any>;
  };

  let stream_state: "open" | "closed" | "waiting" = "closed";
  let audio_player: HTMLAudioElement;
  let pc: RTCPeerConnection;
  let _webrtc_id = Math.random().toString(36).substring(2);

  let _on_change_cb = (msg: "change" | "tick" | "stopword" | any) => {
    if (msg.type === "end_stream") {
      on_change_cb(msg);
      stream_state = "closed";
      stop(pc);
    } else {
      console.debug("calling on_change_cb with msg", msg);
      on_change_cb(msg);
    }
  };

  const dispatch = createEventDispatcher<{
    tick: undefined;
    error: string;
    play: undefined;
    stop: undefined;
  }>();

  async function start_stream(value: string): Promise<string> {
    if (value === "start_webrtc_stream") {
      stream_state = "waiting";
      _webrtc_id = Math.random().toString(36).substring(2);
      value = _webrtc_id;
      pc = new RTCPeerConnection(rtc_configuration);
      pc.addEventListener("connectionstatechange", async (event) => {
        switch (pc.connectionState) {
          case "connected":
            console.info("connected");
            stream_state = "open";
            dispatch("tick");
            break;
          case "disconnected":
            console.info("closed");
            stop(pc);
            break;
          case "failed":
            stream_state = "closed";
            dispatch("error", "Connection failed!");
            stop(pc);
            break;
          default:
            break;
        }
      });
      let stream = null;
      const timeoutId = setTimeout(() => {
        // @ts-ignore
        on_change_cb({ type: "connection_timeout" });
      }, 10000);

      start(
        stream,
        pc,
        audio_player,
        server.offer,
        _webrtc_id,
        "audio",
        _on_change_cb,
      )
        .then((connection) => {
          clearTimeout(timeoutId);
          pc = connection;
        })
        .catch(() => {
          clearTimeout(timeoutId);
          console.info("catching");
          dispatch("error", "Too many concurrent users. Come back later!");
        });
    }
    return value;
  }

  $: start_stream(value as string).then((val) => {
    value = val;
  });
</script>

<BlockLabel
  {show_label}
  Icon={Music}
  float={false}
  label={label || i18n("audio.audio")}
/>
<audio
  class="standard-player"
  class:hidden={true}
  on:load
  bind:this={audio_player}
  on:ended={() => dispatch("stop")}
  on:play={() => dispatch("play")}
/>
{#if value !== "__webrtc_value__"}
  <div class="audio-container">
    <AudioWave
      audio_source_callback={() => audio_player.srcObject}
      {stream_state}
      {icon}
      {icon_button_color}
      {pulse_color}
      {icon_radius}
    />
  </div>
{/if}
{#if value === "__webrtc_value__"}
  <Empty size="small">
    <Music />
  </Empty>
{/if}

<style>
  .audio-container {
    display: flex;
    height: 100%;
    flex-direction: column;
    justify-content: center;
    align-items: center;
  }

  .standard-player {
    width: 100%;
  }

  .hidden {
    display: none;
  }
</style>

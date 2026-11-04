import { useEffect, useRef, useState } from "react";
import * as THREE from "three";

type Avatar2DProps = {
  text?: string;
  onPlay?: () => void;
};

type Sign = {
  word: string;
  label: string;
  frames: number[][];
};

type SignResponse = {
  input_text: string;
  signs: Sign[];
  unsupported_words: string[];
  supported_sign_count: number;
  message: string;
};

type PlaybackFrame = {
  values: number[];
  word: string;
};

type BoneMesh = THREE.Mesh<
  THREE.CylinderGeometry,
  THREE.MeshStandardMaterial
>;

type JointMesh = THREE.Mesh<
  THREE.SphereGeometry,
  THREE.MeshStandardMaterial
>;

const API = "http://127.0.0.1:8000/api";

const FRAME_DURATION = 80;
const FRAME_SIZE = 258;

const POSE_OFFSET = 0;
const LEFT_HAND_OFFSET = 132;
const RIGHT_HAND_OFFSET = 195;

const POSE_CONNECTIONS: [number, number][] = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],

  [11, 23],
  [12, 24],
  [23, 24],

  [23, 25],
  [25, 27],
  [27, 29],
  [29, 31],
  [27, 31],

  [24, 26],
  [26, 28],
  [28, 30],
  [30, 32],
  [28, 32],
];

const HAND_CONNECTIONS: [number, number][] = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

const BODY_POINTS = [
  0, 11, 12, 13, 14, 15, 16,
  23, 24, 25, 26, 27, 28, 29, 30, 31, 32,
];

const UP = new THREE.Vector3(0, 1, 0);

function createMaterial(
  color: number,
  roughness = 0.55,
  metalness = 0.12
) {
  return new THREE.MeshStandardMaterial({
    color,
    roughness,
    metalness,
  });
}

function createBone(
  radius: number,
  color: number
): BoneMesh {
  const geometry = new THREE.CylinderGeometry(
    radius * 0.82,
    radius,
    1,
    12
  );

  const material = createMaterial(color, 0.42, 0.18);

  const mesh = new THREE.Mesh(geometry, material);

  mesh.castShadow = true;
  mesh.receiveShadow = true;

  return mesh;
}

function createJoint(
  radius: number,
  color: number
): JointMesh {
  const geometry = new THREE.SphereGeometry(
    radius,
    16,
    12
  );

  const material = createMaterial(color, 0.3, 0.2);

  const mesh = new THREE.Mesh(geometry, material);

  mesh.castShadow = true;

  return mesh;
}

function updateBone(
  mesh: BoneMesh,
  start: THREE.Vector3 | null,
  end: THREE.Vector3 | null
) {
  if (!start || !end) {
    mesh.visible = false;
    return;
  }

  const direction = new THREE.Vector3().subVectors(end, start);
  const length = direction.length();

  if (length < 0.0001) {
    mesh.visible = false;
    return;
  }

  mesh.visible = true;

  mesh.position.copy(start).add(end).multiplyScalar(0.5);

  mesh.scale.set(1, length, 1);

  mesh.quaternion.setFromUnitVectors(
    UP,
    direction.normalize()
  );
}

function landmarkPosition(
  values: number[],
  index: number,
  stride: number,
  offset: number,
  visibilityOffset = -1
): THREE.Vector3 | null {
  const base = offset + index * stride;

  const x = values[base];
  const y = values[base + 1];
  const z = values[base + 2];

  if (
    !Number.isFinite(x) ||
    !Number.isFinite(y) ||
    !Number.isFinite(z)
  ) {
    return null;
  }

  if (x === 0 && y === 0) {
    return null;
  }

  if (visibilityOffset >= 0) {
    const visibility =
      values[visibilityOffset + index * 4 + 3];

    if (
      Number.isFinite(visibility) &&
      visibility < 0.15
    ) {
      return null;
    }
  }

  // Convert MediaPipe normalized coordinates into
  // the coordinate system used by the 3D scene.
  return new THREE.Vector3(
    (x - 0.5) * 2.5,
    (0.82 - y) * 3.1,
    -z * 1.8
  );
}

export default function Avatar2D({
  text = "",
  onPlay,
}: Avatar2DProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const framesRef = useRef<PlaybackFrame[]>([]);
  const playingRef = useRef(false);
  const startTimeRef = useRef(0);
  const lastFrameRef = useRef(-1);

  const updateFrameRef = useRef<
    ((values: number[]) => void) | null
  >(null);

  const [loading, setLoading] = useState(false);
  const [unsupported, setUnsupported] = useState<string[]>([]);
  const [status, setStatus] = useState(
    "3D avatar ready. Enter text to play signs."
  );
  const [currentWord, setCurrentWord] = useState("");

  // --------------------------------------------------
  // THREE.JS SCENE
  // --------------------------------------------------

  useEffect(() => {
    const container = containerRef.current;

    if (!container) return;

    const scene = new THREE.Scene();

    scene.background = new THREE.Color("#0b1020");

    const camera = new THREE.PerspectiveCamera(
      38,
      1,
      0.1,
      100
    );

    camera.position.set(0, 1.35, 6.2);
    camera.lookAt(0, 1.15, 0);

    const renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: false,
    });

    renderer.setPixelRatio(
      Math.min(window.devicePixelRatio, 2)
    );

    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;

    renderer.outputColorSpace = THREE.SRGBColorSpace;

    container.appendChild(renderer.domElement);

    // --------------------------------------------------
    // LIGHTING
    // --------------------------------------------------

    const ambientLight = new THREE.HemisphereLight(
      0xcbd5ff,
      0x101827,
      2.0
    );

    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(
      0xffffff,
      3
    );

    keyLight.position.set(3, 5, 5);
    keyLight.castShadow = true;

    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(
      0x38bdf8,
      1.3
    );

    fillLight.position.set(-4, 2, 3);

    scene.add(fillLight);

    const rimLight = new THREE.PointLight(
      0x6366f1,
      2,
      12
    );

    rimLight.position.set(0, 3, -3);

    scene.add(rimLight);

    // --------------------------------------------------
    // AVATAR ROOT
    // --------------------------------------------------

    const avatar = new THREE.Group();

    scene.add(avatar);

    // --------------------------------------------------
    // FLOOR
    // --------------------------------------------------

    const floorGeometry = new THREE.PlaneGeometry(
      12,
      12
    );

    const floorMaterial = new THREE.MeshStandardMaterial({
      color: "#111827",
      roughness: 0.9,
      metalness: 0.05,
    });

    const floor = new THREE.Mesh(
      floorGeometry,
      floorMaterial
    );

    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.15;
    floor.position.z = -0.4;

    floor.receiveShadow = true;

    scene.add(floor);

    const grid = new THREE.GridHelper(
      8,
      20,
      0x334155,
      0x1e293b
    );

    grid.position.y = -0.14;

    scene.add(grid);

    // --------------------------------------------------
    // MATERIALS
    // --------------------------------------------------

    const bodyMaterial = createMaterial(
      0x475569,
      0.48,
      0.2
    );

    const armMaterial = createMaterial(
      0x64748b,
      0.35,
      0.22
    );

    const handMaterial = createMaterial(
      0x38bdf8,
      0.3,
      0.15
    );

    const headMaterial = createMaterial(
      0x94a3b8,
      0.35,
      0.12
    );

    // --------------------------------------------------
    // BODY MESHES
    // --------------------------------------------------

    const torso = new THREE.Mesh(
      new THREE.CylinderGeometry(
        0.34,
        0.25,
        1,
        24
      ),
      bodyMaterial
    );

    torso.castShadow = true;
    torso.receiveShadow = true;

    avatar.add(torso);

    const pelvis = new THREE.Mesh(
      new THREE.SphereGeometry(0.28, 20, 16),
      bodyMaterial
    );

    pelvis.scale.set(1.35, 0.72, 0.7);
    pelvis.castShadow = true;

    avatar.add(pelvis);

    const neck = new THREE.Mesh(
      new THREE.CylinderGeometry(
        0.09,
        0.12,
        0.3,
        16
      ),
      bodyMaterial
    );

    neck.castShadow = true;

    avatar.add(neck);

    const head = new THREE.Mesh(
      new THREE.SphereGeometry(0.25, 32, 24),
      headMaterial
    );

    head.scale.set(0.86, 1.15, 0.9);
    head.castShadow = true;

    avatar.add(head);

    // Simple neutral face details.
    const eyeMaterial = createMaterial(0x0f172a);

    const eyeGeometry = new THREE.SphereGeometry(
      0.025,
      12,
      10
    );

    const leftEye = new THREE.Mesh(
      eyeGeometry,
      eyeMaterial
    );

    const rightEye = new THREE.Mesh(
      eyeGeometry,
      eyeMaterial
    );

    leftEye.position.set(-0.075, 0.035, 0.215);
    rightEye.position.set(0.075, 0.035, 0.215);

    head.add(leftEye, rightEye);

    // --------------------------------------------------
    // POSE BONES
    // --------------------------------------------------

    const poseBones = POSE_CONNECTIONS.map(
      ([from, to]) => {
        const radius =
          from === 11 || from === 12 ||
          from === 23 || from === 24
            ? 0.075
            : 0.055;

        const bone = createBone(
          radius,
          0x64748b
        );

        avatar.add(bone);

        return { from, to, mesh: bone };
      }
    );

    // --------------------------------------------------
    // POSE JOINTS
    // --------------------------------------------------

    const poseJoints = BODY_POINTS.map(index => {
      const joint = createJoint(
        index === 0 ? 0.045 : 0.075,
        0x94a3b8
      );

      avatar.add(joint);

      return { index, mesh: joint };
    });

    // --------------------------------------------------
    // HAND BONES
    // --------------------------------------------------

    const createHandBones = (color: number) => {
      return HAND_CONNECTIONS.map(([from, to]) => {
        const bone = createBone(0.025, color);

        avatar.add(bone);

        return { from, to, mesh: bone };
      });
    };

    const leftHandBones = createHandBones(0x38bdf8);
    const rightHandBones = createHandBones(0x818cf8);

    const createHandJoints = (color: number) => {
      return Array.from({ length: 21 }, (_, index) => {
        const joint = createJoint(0.033, color);

        avatar.add(joint);

        return { index, mesh: joint };
      });
    };

    const leftHandJoints = createHandJoints(0x7dd3fc);
    const rightHandJoints = createHandJoints(0xa5b4fc);

    // --------------------------------------------------
    // FRAME UPDATE
    // --------------------------------------------------

    const updateFrame = (values: number[]) => {
      if (values.length !== FRAME_SIZE) return;

      const pose: Array<THREE.Vector3 | null> = [];

      for (let i = 0; i < 33; i++) {
        pose.push(
          landmarkPosition(
            values,
            i,
            4,
            POSE_OFFSET,
            POSE_OFFSET
          )
        );
      }

      const leftHand: Array<THREE.Vector3 | null> = [];

      const rightHand: Array<THREE.Vector3 | null> = [];

      for (let i = 0; i < 21; i++) {
        leftHand.push(
          landmarkPosition(
            values,
            i,
            3,
            LEFT_HAND_OFFSET
          )
        );

        rightHand.push(
          landmarkPosition(
            values,
            i,
            3,
            RIGHT_HAND_OFFSET
          )
        );
      }
      const alignHandToWrist = (
  hand: Array<THREE.Vector3 | null>,
  poseWrist: THREE.Vector3 | null
) => {
  const handWrist = hand[0];

  if (!handWrist || !poseWrist) return;

  const correction = new THREE.Vector3()
    .subVectors(poseWrist, handWrist);

  for (const point of hand) {
    if (point) {
      point.add(correction);
    }
  }
};

alignHandToWrist(leftHand, pose[15]);
alignHandToWrist(rightHand, pose[16]);

      // Update pose bones.
      for (const item of poseBones) {
        updateBone(
          item.mesh,
          pose[item.from],
          pose[item.to]
        );
      }

      // Update pose joints.
      for (const item of poseJoints) {
        const point = pose[item.index];

        item.mesh.visible = point !== null;

        if (point) {
          item.mesh.position.copy(point);
        }
      }

      // Update hand bones.
      const updateHandBones = (
        bones: typeof leftHandBones,
        points: Array<THREE.Vector3 | null>
      ) => {
        for (const item of bones) {
          updateBone(
            item.mesh,
            points[item.from],
            points[item.to]
          );
        }
      };

      updateHandBones(leftHandBones, leftHand);
      updateHandBones(rightHandBones, rightHand);

      // Update hand joints.
      const updateHandJoints = (
        joints: typeof leftHandJoints,
        points: Array<THREE.Vector3 | null>
      ) => {
        for (const item of joints) {
          const point = points[item.index];

          item.mesh.visible = point !== null;

          if (point) {
            item.mesh.position.copy(point);
          }
        }
      };

      updateHandJoints(leftHandJoints, leftHand);
      updateHandJoints(rightHandJoints, rightHand);

      // Torso follows the shoulder and hip landmarks.
      const leftShoulder = pose[11];
      const rightShoulder = pose[12];
      const leftHip = pose[23];
      const rightHip = pose[24];

      if (
        leftShoulder &&
        rightShoulder &&
        leftHip &&
        rightHip
      ) {
        const shoulderCenter = leftShoulder
          .clone()
          .add(rightShoulder)
          .multiplyScalar(0.5);

        const hipCenter = leftHip
          .clone()
          .add(rightHip)
          .multiplyScalar(0.5);

        const torsoDirection = new THREE.Vector3()
          .subVectors(shoulderCenter, hipCenter);

        const torsoLength = torsoDirection.length();

        torso.position
          .copy(shoulderCenter)
          .add(hipCenter)
          .multiplyScalar(0.5);

        torso.scale.set(
          1,
          Math.max(torsoLength, 0.3),
          1
        );

        torso.quaternion.setFromUnitVectors(
          UP,
          torsoDirection.normalize()
        );

        torso.visible = true;

        pelvis.position.copy(hipCenter);

        neck.position
          .copy(shoulderCenter)
          .add(new THREE.Vector3(0, 0.15, 0));

        const nose = pose[0];

        if (nose) {
          head.position.copy(nose);
          head.position.y += 0.23;
        } else {
          head.position.copy(shoulderCenter);
          head.position.y += 0.48;
        }

        neck.visible = true;
        pelvis.visible = true;
        head.visible = true;
      } else {
        torso.visible = false;
        pelvis.visible = false;
        neck.visible = false;
        head.visible = false;
      }
    };

    updateFrameRef.current = updateFrame;

    // --------------------------------------------------
    // RESIZE
    // --------------------------------------------------

    const resize = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;

      if (!width || !height) return;

      camera.aspect = width / height;
      camera.updateProjectionMatrix();

      renderer.setSize(width, height);
    };

    const resizeObserver = new ResizeObserver(resize);

    resizeObserver.observe(container);

    resize();

    // --------------------------------------------------
    // RENDER AND PLAYBACK LOOP
    // --------------------------------------------------

    let renderAnimationId = 0;
    let completed = false;

    const renderLoop = (time: number) => {
      renderAnimationId = requestAnimationFrame(renderLoop);

      if (
        playingRef.current &&
        framesRef.current.length > 0
      ) {
        const elapsed = time - startTimeRef.current;

        const index = Math.floor(
          elapsed / FRAME_DURATION
        );

        if (index >= framesRef.current.length) {
          playingRef.current = false;

          const lastFrame =
            framesRef.current[framesRef.current.length - 1];

          if (lastFrame) {
            updateFrame(lastFrame.values);
            setCurrentWord(lastFrame.word);
          }

          if (!completed) {
            completed = true;
            setStatus("3D sign animation completed.");
          }
        } else if (index !== lastFrameRef.current) {
          lastFrameRef.current = index;

          const frame = framesRef.current[index];

          if (frame) {
            updateFrame(frame.values);
            setCurrentWord(frame.word);
          }
        }
      }

      renderer.render(scene, camera);
    };

    renderAnimationId = requestAnimationFrame(renderLoop);

    // --------------------------------------------------
    // CLEANUP
    // --------------------------------------------------

    return () => {
      cancelAnimationFrame(renderAnimationId);

      resizeObserver.disconnect();

      playingRef.current = false;

      updateFrameRef.current = null;

      scene.traverse(object => {
        if (object instanceof THREE.Mesh) {
          object.geometry.dispose();

          if (Array.isArray(object.material)) {
            object.material.forEach(material => material.dispose());
          } else {
            object.material.dispose();
          }
        }
      });

      renderer.dispose();

      if (renderer.domElement.parentElement === container) {
        container.removeChild(renderer.domElement);
      }
    };
  }, []);

  // --------------------------------------------------
  // STOP PLAYBACK
  // --------------------------------------------------

  const stopPlayback = () => {
    playingRef.current = false;
    lastFrameRef.current = -1;
  };

  // --------------------------------------------------
  // FETCH SIGN SEQUENCES
  // --------------------------------------------------

  const playSigns = async () => {
    const requestedText = text.trim();

    if (!requestedText) {
      stopPlayback();

      framesRef.current = [];

      setCurrentWord("");
      setStatus("Please enter a word or message first.");

      return;
    }

    stopPlayback();

    framesRef.current = [];

    setLoading(true);
    setUnsupported([]);
    setCurrentWord("");

    setStatus("Loading recorded ISL sign sequences...");

    try {
      const response = await fetch(`${API}/avatar/signs`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: requestedText,
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Could not load ISL movements."
        );
      }

      const result = data as SignResponse;

      setUnsupported(result.unsupported_words || []);

      const playbackFrames: PlaybackFrame[] = [];

      for (const sign of result.signs || []) {
        for (const values of sign.frames || []) {
          if (
            Array.isArray(values) &&
            values.length === FRAME_SIZE &&
            values.every(Number.isFinite)
          ) {
            playbackFrames.push({
              values,
              word: sign.word,
            });
          }
        }
      }

      if (playbackFrames.length === 0) {
        setStatus(
          result.message || "No supported sign sequences found."
        );

        return;
      }

      framesRef.current = playbackFrames;

      lastFrameRef.current = -1;
      startTimeRef.current = performance.now();

      playingRef.current = true;

      setStatus(
        `Playing ${result.supported_sign_count ?? result.signs.length} supported sign(s) in 3D.`
      );

      onPlay?.();
    } catch (error) {
      console.error("3D avatar playback error:", error);

      setStatus(
        error instanceof Error
          ? error.message
          : "Backend connection failed."
      );
    } finally {
      setLoading(false);
    }
  };

  // --------------------------------------------------
  // UI
  // --------------------------------------------------

  return (
    <div
      style={{
        width: "100%",
        maxWidth: 650,
        margin: "0 auto",
        padding: 22,
        boxSizing: "border-box",
        borderRadius: 18,
        background: "#0f172a",
        border: "1px solid #334155",
        boxShadow: "0 20px 35px rgba(0,0,0,0.35)",
        textAlign: "center",
        fontFamily: "system-ui, sans-serif",
      }}
    >
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 14,
        }}
      >
        <span
          style={{
            fontSize: 12,
            fontWeight: 800,
            color: "#38bdf8",
            letterSpacing: 1,
          }}
        >
          UNICOMM • 3D AVATAR
        </span>

        <span
          style={{
            fontSize: 11,
            color: loading ? "#fbbf24" : "#94a3b8",
          }}
        >
          {loading
            ? "LOADING"
            : playingRef.current
              ? "PLAYING"
              : "READY"}
        </span>
      </div>

      <div
        ref={containerRef}
        role="img"
        aria-label="Three-dimensional animated ISL skeletal avatar"
        style={{
          width: "100%",
          height: 440,
          overflow: "hidden",
          borderRadius: 14,
          border: "1px solid #334155",
          background: "#0b1020",
        }}
      />

      <div
        style={{
          marginTop: 15,
          padding: 12,
          borderRadius: 9,
          background: "#090d16",
          border: "1px solid #334155",
        }}
      >
        <p
          style={{
            color: "#f8fafc",
            fontSize: 14,
            margin: 0,
            overflowWrap: "anywhere",
          }}
        >
          {text.trim() || "Awaiting target text..."}
        </p>
      </div>

      {currentWord && (
        <p
          style={{
            color: "#38bdf8",
            fontSize: 13,
            fontWeight: 600,
            marginTop: 10,
          }}
        >
          Active sign: {currentWord}
        </p>
      )}

      <div
        style={{
          display: "flex",
          gap: 10,
          marginTop: 16,
        }}
      >
        <button
          type="button"
          disabled={loading}
          onClick={playSigns}
          style={{
            flex: 1,
            padding: 13,
            border: 0,
            borderRadius: 9,
            background: loading ? "#475569" : "#0284c7",
            color: "#fff",
            fontWeight: 700,
            cursor: loading ? "wait" : "pointer",
          }}
        >
          {loading ? "Loading..." : "Play 3D ISL signs"}
        </button>

        <button
          type="button"
          onClick={() => {
            stopPlayback();
            setStatus("Playback stopped.");
          }}
          style={{
            padding: "13px 20px",
            border: "1px solid #475569",
            borderRadius: 9,
            background: "#1e293b",
            color: "#fff",
            fontWeight: 600,
            cursor: "pointer",
          }}
        >
          Stop
        </button>
      </div>

      <p
        role="status"
        aria-live="polite"
        style={{
          color: "#94a3b8",
          fontSize: 12,
          marginTop: 13,
        }}
      >
        {status}
      </p>

      {unsupported.length > 0 && (
        <p
          style={{
            color: "#f87171",
            fontSize: 12,
            overflowWrap: "anywhere",
          }}
        >
          Unsupported words: {unsupported.join(", ")}
        </p>
      )}

      <small
        style={{
          display: "block",
          marginTop: 12,
          paddingTop: 10,
          borderTop: "1px solid #334155",
          color: "#64748b",
          fontSize: 11,
        }}
      >
        Three.js • 3D skeletal rendering • AMRITA sign sequences
      </small>
    </div>
  );
}
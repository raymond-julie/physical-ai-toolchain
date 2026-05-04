import type { JointGroup } from './joint-constants'
import { JOINT_COLORS } from './joint-constants'
import { JointSelector } from './JointSelector'

export type TrajectoryPlotMode = 'position' | 'velocity' | 'action'

interface TrajectoryPlotControlsProps {
  jointCount: number
  selectedJoints: number[]
  onSelectJoints: (joints: number[]) => void
  groups: JointGroup[]
  labels: Record<string, string>
  onEditJointLabel: (jointIndex: number, label: string) => void
  onEditGroupLabel: (groupId: string, label: string) => void
  onCreateGroup: (label: string, joints: number[]) => void
  onDeleteGroup: (groupId: string) => void
  onMoveJoint: (
    jointIndex: number,
    sourceGroupId: string,
    targetGroupId: string,
    toPosition: number,
  ) => void
  onOpenDefaults: () => void
  mode: TrajectoryPlotMode
  onSetMode: (mode: TrajectoryPlotMode) => void
  showNormalized: boolean
  isNormalizationDisabled: boolean
  onToggleNormalization: () => void
  selectorEditable?: boolean
}

export function TrajectoryPlotControls({
  jointCount,
  selectedJoints,
  onSelectJoints,
  groups,
  labels,
  onEditJointLabel,
  onEditGroupLabel,
  onCreateGroup,
  onDeleteGroup,
  onMoveJoint,
  onOpenDefaults,
  mode,
  onSetMode,
  showNormalized,
  isNormalizationDisabled,
  onToggleNormalization,
  selectorEditable = true,
}: TrajectoryPlotControlsProps) {
  return (
    <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
      <div
        data-testid="trajectory-joint-selector-scroll"
        className="max-h-40 w-full min-w-0 flex-1 overflow-y-auto pr-2 lg:max-h-32"
      >
        <JointSelector
          jointCount={jointCount}
          selectedJoints={selectedJoints}
          onSelectJoints={onSelectJoints}
          colors={JOINT_COLORS}
          groups={groups}
          labels={labels}
          editable={selectorEditable}
          onEditJointLabel={selectorEditable ? onEditJointLabel : undefined}
          onEditGroupLabel={selectorEditable ? onEditGroupLabel : undefined}
          onCreateGroup={selectorEditable ? onCreateGroup : undefined}
          onDeleteGroup={selectorEditable ? onDeleteGroup : undefined}
          onMoveJoint={selectorEditable ? onMoveJoint : undefined}
          onOpenDefaults={selectorEditable ? onOpenDefaults : undefined}
        />
      </div>
      <div className="flex w-full shrink-0 flex-wrap items-center gap-2 lg:w-auto lg:justify-end lg:self-start">
        <button
          onClick={() => onSetMode('position')}
          className={
            mode === 'position'
              ? 'bg-primary text-primary-foreground rounded-sm px-2 py-1 text-xs'
              : 'bg-muted text-muted-foreground rounded-sm px-2 py-1 text-xs'
          }
        >
          Position
        </button>
        <button
          onClick={() => onSetMode('velocity')}
          className={
            mode === 'velocity'
              ? 'bg-primary text-primary-foreground rounded-sm px-2 py-1 text-xs'
              : 'bg-muted text-muted-foreground rounded-sm px-2 py-1 text-xs'
          }
        >
          Velocity
        </button>
        <button
          onClick={() => onSetMode('action')}
          className={
            mode === 'action'
              ? 'bg-primary text-primary-foreground rounded-sm px-2 py-1 text-xs'
              : 'bg-muted text-muted-foreground rounded-sm px-2 py-1 text-xs'
          }
        >
          Action
        </button>
        <button
          type="button"
          aria-pressed={showNormalized}
          aria-disabled={isNormalizationDisabled}
          disabled={isNormalizationDisabled}
          onClick={onToggleNormalization}
          className={
            isNormalizationDisabled
              ? 'bg-muted text-muted-foreground/60 cursor-not-allowed rounded-sm border border-transparent px-2 py-1 text-xs transition-colors'
              : showNormalized
                ? 'border-primary bg-primary text-primary-foreground rounded-sm border px-2 py-1 text-xs transition-colors'
                : 'bg-muted text-muted-foreground hover:border-border rounded-sm border border-transparent px-2 py-1 text-xs transition-colors'
          }
        >
          Normalize
        </button>
      </div>
    </div>
  )
}

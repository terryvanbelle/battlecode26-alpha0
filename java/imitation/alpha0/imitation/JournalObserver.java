package alpha0.imitation;

import battlecode.common.RobotController;
import java.io.*;
import java.nio.file.*;
import java.util.*;
import java.util.function.IntSupplier;

/** Per-robot observer. Lifecycle reasons and physical hashes must come from the harness. */
public final class JournalObserver implements PrimitiveRecorder.Observer {
    private final TraceJournal journal;
    private final ObservationStore observations;
    private final IntSupplier budget;
    private int turn;
    public JournalObserver(TraceJournal journal,ObservationStore observations,IntSupplier budget) {
        this.journal=Objects.requireNonNull(journal);this.observations=Objects.requireNonNull(observations);
        this.budget=Objects.requireNonNull(budget);
    }
    /** Only PublicObservation.capture output enters this store, never world/teacher objects. */
    public static final class ObservationStore {
        private final Path directory;
        private final long limit;
        private long bytes;
        private final Set<String> hashes=new HashSet<>();
        public ObservationStore(Path directory,long maxBytes)throws IOException {
            if(maxBytes<=0)throw new IllegalArgumentException("invalid observation cap");
            this.directory=directory;limit=maxBytes;Files.createDirectory(directory);
        }
        synchronized String capture(RobotController rc)throws IOException {
            Object publicData;
            try{publicData=PublicObservation.capture(rc);}
            catch(battlecode.common.GameActionException e){throw new IOException("public observation failed",e);}
            byte[] encoded=TraceJournal.canonical(publicData);
            String sha=TraceJournal.digest(publicData);
            if(!hashes.contains(sha)) {
                if(encoded.length>1024*1024 || encoded.length>limit-bytes)throw new IOException("observation capacity exceeded");
                Files.write(directory.resolve(sha+".json"),encoded,StandardOpenOption.CREATE_NEW,StandardOpenOption.WRITE);
                bytes+=encoded.length;hashes.add(sha);
            }
            return sha;
        }
    }
    public PrimitiveRecorder.Snapshot before(RobotController rc) {
        try {
            int start=bytecodes(rc);String sha=observations.capture(rc);
            if(bytecodes(rc)!=start)throw new IllegalStateException("observation capture changed charged budget");
            return new PrimitiveRecorder.Snapshot(sha,start);
        }catch(IOException e){journal.invalidate();throw new UncheckedIOException(e);}
        catch(RuntimeException e){journal.invalidate();throw e;}
    }
    public int bytecodes(RobotController rc) {
        int n=budget.getAsInt();if(n<0)throw new IllegalStateException("invalid budget");return n;
    }
    public void accept(PrimitiveRecorder.Event event) {
        Map<String,Object> p=new TreeMap<>();p.put("operation",event.operation());p.put("arguments",event.arguments());
        p.put("outcome",event.exceptionClass()==null?"returned":"exception");p.put("return_boolean",event.returnedBoolean());
        p.put("exception",event.exceptionClass());p.put("public_observation_sha256",event.before().publicObservationSha256());
        p.put("bytecodes_before",event.before().bytecodes());p.put("bytecodes_after",event.bytecodesAfter());
        try{journal.append(event.robotId(),event.team().name(),event.round(),turn,"action",p);}
        catch(IOException e){throw new UncheckedIOException(e);}
    }
    /** No automatic inference that a pause means yield or that detach means death. */
    public void endTurn(RobotController rc,String physicalAuditHash,String termination,int chargedBytecodes) {
        TraceJournal.hash(physicalAuditHash);
        if(!Set.of("yield","death","resign","overrun","exception").contains(termination)||chargedBytecodes<0)
            throw new IllegalArgumentException("invalid turn proof");
        PrimitiveRecorder.Snapshot observation=before(rc);
        Map<String,Object> p=Map.of("public_observation_sha256",observation.publicObservationSha256(),
            "physical_state_sha256",physicalAuditHash,"bytecodes_used",chargedBytecodes,"termination",termination);
        try{journal.append(rc.getID(),rc.getTeam().name(),rc.getRoundNum(),turn,"turn_end",p);turn++;}
        catch(IOException e){throw new UncheckedIOException(e);}
    }
}

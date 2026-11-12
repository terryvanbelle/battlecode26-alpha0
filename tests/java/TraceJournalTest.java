import alpha0.imitation.TraceJournal;
import java.nio.file.*;
import java.util.*;
public class TraceJournalTest {
 public static void main(String[] args)throws Exception {
  Path dir=Path.of(args[0]);String h="a".repeat(64);
  Map<String,Object> action=new TreeMap<>();action.put("operation","squeak");action.put("arguments",List.of(42));
  action.put("outcome","returned");action.put("return_boolean",false);action.put("exception",null);
  action.put("public_observation_sha256",h);action.put("bytecodes_before",10);action.put("bytecodes_after",11);
  Map<String,Object> end=Map.of("public_observation_sha256",h,"physical_state_sha256",h,"bytecodes_used",12,"termination","yield");
  try(var j=new TraceJournal(dir.resolve("trace.jsonl"),h,10000)) {
   j.append(1,"A",1,0,"action",action);j.append(1,"A",1,0,"turn_end",end);
   j.seal(dir.resolve("seal.json"),h,h,h,h);
  }
  try {new TraceJournal(dir.resolve("trace.jsonl"),h,10000);throw new AssertionError("overwrite");}catch(FileAlreadyExistsException expected){}
  try(var j=new TraceJournal(dir.resolve("capped.jsonl"),h,10)) {
   try{j.append(1,"A",1,0,"action",action);throw new AssertionError("cap");}catch(java.io.IOException expected){}
   try{j.seal(dir.resolve("bad-seal.json"),h,h,h,h);throw new AssertionError("failed seal");}catch(IllegalStateException expected){}
  }
  Files.writeString(dir.resolve("unicode-hash.txt"),TraceJournal.digest(Map.of("s","é😀\n","n",-1)));
  System.out.println("PASS bounded journal, exclusive creation, failed seal, Unicode canonical hash");
 }
}

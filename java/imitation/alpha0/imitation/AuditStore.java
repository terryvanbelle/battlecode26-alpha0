package alpha0.imitation;
import java.nio.file.*;
import java.io.*;
import java.security.*;
import java.util.*;
import java.util.zip.DeflaterOutputStream;
/** Lossless bounded content-addressed storage; hashes describe uncompressed bytes. */
public final class AuditStore {
 private final Path root;private final long limit;private long bytes;
 public AuditStore(Path root,long limit)throws IOException{this.root=root;this.limit=limit;Files.createDirectory(root);Files.createDirectory(root.resolve("chunks"));}
 private static String hash(byte[] b){try{return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(b));}catch(NoSuchAlgorithmException e){throw new IllegalStateException(e);}}
 private void write(Path path,byte[] data)throws IOException{if(data.length>limit-bytes)throw new IOException("audit storage cap");Files.write(path,data,StandardOpenOption.CREATE_NEW);bytes+=data.length;}
 public synchronized String put(byte[] data)throws IOException{
  if(data.length>16*1024*1024)throw new IOException("audit snapshot cap");
  String sha=hash(data);Path index=root.resolve(sha+".index.json");if(Files.exists(index))return sha;
  List<String> chunks=new ArrayList<>();
  for(int i=0;i<data.length;i+=4096){byte[] part=Arrays.copyOfRange(data,i,Math.min(i+4096,data.length));String id=hash(part);chunks.add(id);Path path=root.resolve("chunks").resolve(id+".zlib");
   if(!Files.exists(path)){var buffer=new ByteArrayOutputStream();try(var zip=new DeflaterOutputStream(buffer)){zip.write(part);}write(path,buffer.toByteArray());}
  }
  write(index,TraceJournal.canonical(Map.of("schema","audit-chunks-v1","sha256",sha,"length",data.length,"chunk_size",4096,"chunks",chunks)));return sha;
 }
}
